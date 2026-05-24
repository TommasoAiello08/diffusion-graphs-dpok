"""
SG-Aware reward — BLIP-ITM-only, graph-structured.

No OwlViT, no CLIP, no region cropping. The reward iterates over the
graph's nodes and edges and scores each piece with BLIP-ITM on the
whole image, using antonym contrast where applicable.

Pipeline per (image, scene_graph):
  - Object node     → BLIP-ITM("a photo of a <obj>") — absolute P(match)
  - Attribute edge  → BLIP-ITM("a <attr> <obj>")  − BLIP-ITM("a <antonym> <obj>")
  - Relation edge   → BLIP-ITM("<s> <r> <o>")     − BLIP-ITM("<s> <antonym> <o>")
  - Global          → ImageReward(image, stringified_graph)   [small weight]

Sensitivity by design:
  - remove_object   → that node's BLIP-ITM term disappears (mean drops)
  - swap_attribute  → antonym contrast flips sign
  - swap_spatial    → antonym contrast on spatial language (weaker than bbox,
                      but only neural option without a detector)
  - swap_noun       → object's BLIP-ITM("a photo of a <distractor>") low

Trade vs the previous OwlViT version: lose bbox-geometry spatial predicate
(was a clean 1/0 flip) and lose region-cropped attribute scoring (now full
image). Antonym contrast is still the dominant discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from dpok_scene_reward import SceneGraph, BLIPRewardModel
from perturb_graph import (
    ATTRIBUTE_ANTONYMS,
    SPATIAL_RELATIONS,
    stringify_graph,
)


# ─────────────────────────────────────────────────────────────────────────────
# Relation antonyms for non-spatial (used in BLIP-ITM contrast)
# ─────────────────────────────────────────────────────────────────────────────

NON_SPATIAL_RELATION_ANTONYMS = {
    "holding": "ignoring",
    "wearing": "without",
    "riding": "standing beside",
    "eating": "ignoring",
    "looking at": "looking away from",
    "touching": "far from",
    "carrying": "dropping",
    "playing with": "ignoring",
    "drinking": "ignoring",
    "pulling": "pushing", "pushing": "pulling",
    "kissing": "ignoring",
    "hugging": "ignoring",
    "sitting on": "standing beside",
    "on top of": "underneath",
    "underneath": "on top of",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SGAwareConfig:
    w_obj:    float = 0.40
    w_attr:   float = 0.30
    w_rel:    float = 0.20
    w_global: float = 0.10
    blip_model_id: str = "Salesforce/blip-itm-base-coco"
    use_global_imagereward: bool = True
    # Bias correction for object presence: subtract this so P(match) for a
    # plausible "a photo of an X" is centred near zero.
    object_baseline: float = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Main reward
# ─────────────────────────────────────────────────────────────────────────────

class SGAwareReward:
    """BLIP-ITM-only graph-structured reward.

    Usage:
        scorer = SGAwareReward(device='cuda')
        reward, breakdown = scorer.score(pil_image, scene_graph)
    """

    def __init__(self, device: str = "cuda",
                 config: Optional[SGAwareConfig] = None,
                 image_reward_model=None,
                 blip_reward_model: Optional[BLIPRewardModel] = None):
        self.device = device
        self.cfg    = config or SGAwareConfig()

        # BLIP-ITM — either reuse an existing instance or load our own
        if blip_reward_model is not None:
            self.blip = blip_reward_model
            print("[SG] Reusing externally-provided BLIPRewardModel.")
        else:
            print(f"[SG] Loading BLIP-ITM ({self.cfg.blip_model_id}) ...")
            self.blip = BLIPRewardModel(
                device=device, baseline=0.0, scale=1.0,
            )

        # ImageReward — optional global term, may be injected
        self.image_reward_model = image_reward_model
        if self.cfg.use_global_imagereward and self.image_reward_model is None:
            try:
                import ImageReward as RM
                print("[SG] Loading ImageReward-v1.0 for global regularizer ...")
                self.image_reward_model = RM.load(
                    "ImageReward-v1.0", device=device)
            except Exception as e:
                print(f"[SG] ImageReward unavailable ({e}); w_global = 0.")
                self.cfg.w_global = 0.0

        print("[SG] Ready (BLIP-only, no detector).")

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _p_match(self, image: Image.Image, text: str) -> float:
        """Return P(image matches text) ∈ [0, 1] via BLIP-ITM head.

        Goes through BLIPRewardModel.compute which returns
        (p_match - baseline) * scale. We constructed BLIPRewardModel with
        baseline=0, scale=1, so the returned value is the raw P(match).
        """
        return float(self.blip.compute(image, text).cpu())

    def _antonym_for_attr(self, attr: str) -> Optional[str]:
        return ATTRIBUTE_ANTONYMS.get(attr.lower())

    def _antonym_for_relation(self, rel: str) -> Optional[str]:
        r = rel.lower()
        if r in SPATIAL_RELATIONS:
            return SPATIAL_RELATIONS[r]
        return NON_SPATIAL_RELATION_ANTONYMS.get(r)

    # ──────────────────────────────────────────────────────────────────────
    # Main score
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(self, image: Image.Image,
              graph: SceneGraph) -> Tuple[float, Dict]:
        image = image.convert("RGB")
        breakdown: Dict = {
            "objects": {}, "attributes": [], "relations": [], "global": None,
        }

        # 1. Object presence — BLIP-ITM("a photo of an X") with baseline-subtract
        obj_scores: List[float] = []
        for o in graph.objects:
            p = self._p_match(image, f"a photo of a {o}")
            s = p - self.cfg.object_baseline
            obj_scores.append(s)
            breakdown["objects"][o] = {"p_match": p, "score": s}
        obj_mean = (sum(obj_scores) / len(obj_scores)) if obj_scores else 0.0

        # 2. Per-attribute contrast (full image)
        attr_scores: List[float] = []
        for (obj, attr) in graph.attributes:
            p_pos = self._p_match(image, f"a {attr} {obj}")
            antonym = self._antonym_for_attr(attr)
            if antonym is None:
                s = 2.0 * (p_pos - 0.5)
                p_neg = None
            else:
                p_neg = self._p_match(image, f"a {antonym} {obj}")
                s = p_pos - p_neg
            attr_scores.append(s)
            breakdown["attributes"].append(
                {"obj": obj, "attr": attr, "antonym": antonym,
                 "p_pos": p_pos, "p_neg": p_neg, "score": s},
            )
        attr_mean = (sum(attr_scores) / len(attr_scores)) if attr_scores else 0.0

        # 3. Per-relation contrast (full image — no detector, no crop)
        rel_scores: List[float] = []
        for (subj, rel, obj) in graph.relations:
            p_pos = self._p_match(image, f"{subj} {rel} {obj}")
            antonym = self._antonym_for_relation(rel)
            if antonym is None:
                s = 2.0 * (p_pos - 0.5)
                p_neg = None
            else:
                p_neg = self._p_match(image, f"{subj} {antonym} {obj}")
                s = p_pos - p_neg
            rel_scores.append(s)
            breakdown["relations"].append(
                {"triple": (subj, rel, obj), "antonym": antonym,
                 "p_pos": p_pos, "p_neg": p_neg, "score": s,
                 "kind": "blip-contrast"},
            )
        rel_mean = (sum(rel_scores) / len(rel_scores)) if rel_scores else 0.0

        # 4. Global regularizer = ImageReward on stringified graph
        global_score = 0.0
        if (self.cfg.use_global_imagereward
                and self.cfg.w_global > 0
                and self.image_reward_model is not None):
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(tmp.name); tmp.close()
            try:
                global_score = float(
                    self.image_reward_model.score(
                        stringify_graph(graph), tmp.name)
                )
            except Exception:
                global_score = 0.0
            finally:
                try: os.unlink(tmp.name)
                except OSError: pass
        breakdown["global"] = global_score

        # 5. Aggregate
        reward = (
            self.cfg.w_obj    * obj_mean
            + self.cfg.w_attr * attr_mean
            + self.cfg.w_rel  * rel_mean
            + self.cfg.w_global * global_score
        )

        breakdown["aggregate"] = {
            "obj_mean": obj_mean, "attr_mean": attr_mean,
            "rel_mean": rel_mean, "global": global_score,
            "reward": reward,
        }
        return reward, breakdown
