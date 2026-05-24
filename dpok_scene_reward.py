"""
Data loading + reward computation for DPOK training.
Self-contained — no src/ imports.

Three data sources:
  COCO captions   → plain text prompts, no scene graph
  PSG scene graphs → rich relational prompts + SceneGraph for eval
  GQA scene graphs → rich attribute+relation prompts + SceneGraph for eval

Prompt generation mirrors generate_prompts_psg_gqa.py but is integrated
here so this file stays fully self-contained.

Reward (training):
  Simple CLIPScore:  reward = (cosine_sim(image, text) - baseline) * scale
  This is intentionally plain — stable and fast.

Evaluation (compute_detailed):
  Decomposed CLIP breakdown: per-object, per-relation, per-attribute scores.
  These are NOT used during training, only in dpok_eval.py.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import open_clip


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SceneGraph:
    """Lightweight scene graph for evaluation decomposition."""
    objects:    List[str]                   # ["person", "car", "dog"]
    attributes: List[Tuple[str, str]]       # [("car", "red"), ("dog", "small")]
    relations:  List[Tuple[str, str, str]]  # [("person", "riding", "car")]


@dataclass
class PromptItem:
    """One training sample: text prompt + optional scene graph."""
    text:           str
    object_text:    str = ""               # object-list version of the prompt
    relation_text:  str = ""               # relation-only version of the prompt
    graph:          Optional[SceneGraph] = None
    source:         str = "unknown"        # "coco" | "psg" | "gqa"
    image_id:       str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_name(name: str) -> str:
    """Remove COCO-style suffixes: 'fence-merged' → 'fence'."""
    for suffix in ("-merged", "-other", "-stuff"):
        name = name.replace(suffix, "")
    return name.strip()


# ─────────────────────────────────────────────────────────────────────────────
# COCO captions
# ─────────────────────────────────────────────────────────────────────────────

def load_coco_items(
    json_path: str,
    max_captions: int = 500,
    seed: int = 42,
) -> List[PromptItem]:
    """Load COCO captions → PromptItems (no scene graph)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    captions = [
        a["caption"].strip()
        for a in data.get("annotations", [])
        if a.get("caption", "").strip()
    ]
    random.Random(seed).shuffle(captions)
    items = [
        PromptItem(
            text=c,
            object_text=c,
            relation_text=c,
            source="coco",
        )
        for c in captions[:max_captions]
    ]
    print(f"[COCO] {len(items)} captions loaded from {json_path}")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# PSG (OpenPSG / Panoptic Scene Graph)
# ─────────────────────────────────────────────────────────────────────────────

def _psg_obj_name(annotations: list, obj_classes: list, idx: int) -> str:
    if idx >= len(annotations):
        return "object"
    cid = annotations[idx].get("category_id", 0)
    if 0 <= cid < len(obj_classes):
        return _clean_name(obj_classes[cid])
    return "object"


def load_psg_items(
    json_path: str,
    max_scenes: int = 500,
    max_rel: int = 8,
    seed: int = 42,
) -> List[PromptItem]:
    """
    Load OpenPSG annotations → PromptItems with SceneGraph.

    Prompt style (mirrors generate_prompts_psg_gqa.py):
      scene_prompt   : "A scene where X pred Y, A pred B."
      object_prompt  : "A photo containing cat, person, bicycle."
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    thing_classes = data.get("thing_classes", [])
    stuff_classes = data.get("stuff_classes", [])
    pred_classes  = data.get("predicate_classes", [])
    obj_classes   = thing_classes + stuff_classes

    scenes = list(data.get("data", []))
    random.Random(seed).shuffle(scenes)

    items = []
    for scene in scenes:
        if len(items) >= max_scenes:
            break

        rels_raw  = scene.get("relations", [])
        anns      = scene.get("annotations", [])
        if not rels_raw or len(anns) < 2:
            continue

        # Build unique object list
        obj_names    = [_psg_obj_name(anns, obj_classes, i) for i in range(len(anns))]
        unique_objs  = list(dict.fromkeys(obj_names))

        # Build relation triples
        relation_triples: List[Tuple[str, str, str]] = []
        rel_phrases: List[str] = []
        for rel in rels_raw[:max_rel]:
            s_idx, o_idx, p_idx = rel[0], rel[1], rel[2]
            sn = _psg_obj_name(anns, obj_classes, s_idx)
            on = _psg_obj_name(anns, obj_classes, o_idx)
            pn = pred_classes[p_idx] if 0 <= p_idx < len(pred_classes) else "near"
            relation_triples.append((sn, pn, on))
            rel_phrases.append(f"{sn} {pn} {on}")

        if len(unique_objs) < 2:
            continue

        # Build prompt texts
        object_text   = "A photo containing " + ", ".join(unique_objs[:10]) + "."
        relation_text = ("A scene where " + ", ".join(rel_phrases) + ".") if rel_phrases \
                        else object_text
        scene_text    = relation_text

        graph = SceneGraph(
            objects=unique_objs[:10],
            attributes=[],           # PSG doesn't have attributes in the same format
            relations=relation_triples,
        )
        items.append(PromptItem(
            text=scene_text,
            object_text=object_text,
            relation_text=relation_text,
            graph=graph,
            source="psg",
            image_id=str(scene.get("image_id", "")),
        ))

    print(f"[PSG] {len(items)} prompts loaded from {json_path}")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# GQA (Visual Genome Question Answering)
# ─────────────────────────────────────────────────────────────────────────────

def _gqa_short_desc(obj: dict) -> str:
    """'<attr1> <attr2> <name>' or just '<name>'."""
    attrs = obj.get("attributes", [])[:2]
    name  = obj.get("name", "object")
    return (" ".join(attrs) + " " + name).strip() if attrs else name


def _gqa_sorted_objects(scene: dict) -> List[Tuple[str, dict]]:
    """Objects sorted by area (largest first)."""
    items = list(scene.get("objects", {}).items())
    items.sort(key=lambda kv: kv[1].get("h", 0) * kv[1].get("w", 0), reverse=True)
    return items


def _gqa_collect_relations(scene: dict) -> List[Tuple[str, str, str]]:
    """(src_id, dst_id, rel_name) triples from a GQA scene."""
    rels = []
    objects = scene.get("objects", {})
    for src_id, obj in objects.items():
        rel_list = obj.get("relations", [])
        if isinstance(rel_list, dict):
            rel_list = list(rel_list.values())
        for rel in rel_list:
            dst_id = str(rel.get("object", ""))
            if dst_id in objects:
                rels.append((str(src_id), dst_id, str(rel.get("name", "near"))))
    return rels


def load_gqa_items(
    json_path: str,
    max_scenes: int = 500,
    max_objects: int = 6,
    max_rel: int = 8,
    seed: int = 42,
) -> List[PromptItem]:
    """
    Load GQA scene graphs → PromptItems with SceneGraph.

    Prompt style (mirrors generate_prompts_psg_gqa.py):
      scene_prompt   : "A scene in a kitchen containing red apple, wooden table
                        where apple on table, person near table."
      object_prompt  : "A photo containing red apple, wooden table, person."
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    image_ids = list(data.keys())
    random.Random(seed).shuffle(image_ids)

    items = []
    for img_id in image_ids:
        if len(items) >= max_scenes:
            break

        scene   = data[img_id]
        objects = scene.get("objects", {})
        if len(objects) < 2:
            continue

        sorted_objs    = _gqa_sorted_objects(scene)[:max_objects]
        id_to_obj      = {oid: obj for oid, obj in sorted_objs}
        all_rel_triples = _gqa_collect_relations(scene)
        if not all_rel_triples:
            continue

        # Object names and descriptions
        unique_names: List[str] = []
        attr_pairs:   List[Tuple[str, str]] = []
        for oid, obj in sorted_objs:
            name = obj.get("name", "object")
            if name not in unique_names:
                unique_names.append(name)
            for attr in obj.get("attributes", [])[:2]:
                attr_pairs.append((name, attr))

        # Relation triples using top objects only
        rel_triples: List[Tuple[str, str, str]] = []
        rel_phrases: List[str] = []
        for src_id, dst_id, rn in all_rel_triples[:max_rel]:
            if src_id in id_to_obj and dst_id in id_to_obj:
                sn = _gqa_short_desc(id_to_obj[src_id])
                dn = _gqa_short_desc(id_to_obj[dst_id])
                # also store plain names in triples for eval
                rel_triples.append((id_to_obj[src_id].get("name", "object"), rn,
                                    id_to_obj[dst_id].get("name", "object")))
                rel_phrases.append(f"{sn} {rn} {dn}")

        if len(unique_names) < 2:
            continue

        # Object-list prompt (with attributes)
        descs        = ", ".join(_gqa_short_desc(obj) for _, obj in sorted_objs)
        object_text  = "A photo containing " + descs + "."

        # Relation prompt
        relation_text = ("; ".join(rel_phrases)) if rel_phrases else object_text

        # Full scene prompt with optional location/weather context
        location = scene.get("location")
        weather  = scene.get("weather")
        prefix   = "A scene"
        ctx = []
        if location:
            ctx.append(f"in a {location}")
        if weather and str(weather).lower() not in ("none", ""):
            ctx.append(f"with {weather} weather")
        if ctx:
            prefix += " " + " ".join(ctx)

        scene_bits = []
        if descs:
            scene_bits.append(f"containing {descs}")
        if rel_phrases:
            scene_bits.append("where " + ", ".join(rel_phrases[:4]))
        scene_text = (prefix + " " + "; ".join(scene_bits) + ".") if scene_bits \
                     else (prefix + ".")
        scene_text = scene_text.replace("..", ".")

        graph = SceneGraph(
            objects=unique_names,
            attributes=attr_pairs,
            relations=rel_triples,
        )
        items.append(PromptItem(
            text=scene_text,
            object_text=object_text,
            relation_text=relation_text,
            graph=graph,
            source="gqa",
            image_id=str(img_id),
        ))

    print(f"[GQA] {len(items)} prompts loaded from {json_path}")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Reward model — simple CLIP
# ─────────────────────────────────────────────────────────────────────────────

class RewardModel:
    """
    CLIP-based reward for DPOK. Self-contained, no src/ dependency.

    Training reward:  reward = (CLIP(image, text) - baseline) * scale
    Evaluation only:  compute_detailed() also returns per-object/relation/attribute
                      decomposed scores for analysis — not used during training.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        baseline: float = 0.20,
        scale: float = 10.0,
    ):
        self.device   = device
        self.baseline = baseline
        self.scale    = scale

        print(f"Loading CLIP reward model ({model_name})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        self.model     = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        print("CLIP reward model loaded.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        img_t = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = F.normalize(self.model.encode_image(img_t), dim=-1)
        return feat

    def _clip_score(self, img_feat: torch.Tensor, text: str) -> float:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            txt_feat = F.normalize(self.model.encode_text(tokens), dim=-1)
        return float((img_feat * txt_feat).sum(dim=-1).cpu())

    def _clip_scores_batch(
        self, img_feat: torch.Tensor, texts: List[str]
    ) -> List[float]:
        if not texts:
            return []
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            txt_feats = F.normalize(self.model.encode_text(tokens), dim=-1)
        scores = (img_feat @ txt_feats.T).squeeze(0)
        return scores.cpu().tolist()

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute(self, pil_image: Image.Image, item: PromptItem) -> torch.Tensor:
        """
        Compute training reward. Returns scalar float32 tensor on self.device.
        Always uses plain global CLIPScore — fast and stable.
        """
        img_feat = self._encode_image(pil_image)
        raw      = self._clip_score(img_feat, item.text)
        reward   = (raw - self.baseline) * self.scale
        return torch.tensor(reward, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def compute_detailed(self, pil_image: Image.Image, item: PromptItem) -> dict:
        """
        Compute full evaluation breakdown.
        Returns a dict with clip_score, and optionally:
          object_scores, object_mean
          relation_scores, relation_mean
          attribute_scores, attribute_mean
        Used by dpok_eval.py — NOT called during training.
        """
        img_feat = self._encode_image(pil_image)
        result   = {
            "clip_score": self._clip_score(img_feat, item.text),
            "source":     item.source,
            "image_id":   item.image_id,
        }

        g = item.graph
        if g is None or len(g.objects) < 2:
            return result

        # Object recall
        obj_texts  = [f"a photo of a {o}" for o in g.objects]
        obj_scores = self._clip_scores_batch(img_feat, obj_texts)
        result["object_scores"] = dict(zip(g.objects, obj_scores))
        result["object_mean"]   = float(sum(obj_scores) / len(obj_scores))

        # Relation score
        if g.relations:
            rel_texts  = [f"{s} {p} {o}" for s, p, o in g.relations]
            rel_scores = self._clip_scores_batch(img_feat, rel_texts)
            rel_labels = [f"{s}-{p}-{o}" for s, p, o in g.relations]
            result["relation_scores"] = dict(zip(rel_labels, rel_scores))
            result["relation_mean"]   = float(sum(rel_scores) / len(rel_scores))

        # Attribute score
        if g.attributes:
            attr_texts  = [f"a {attr} {obj}" for obj, attr in g.attributes]
            attr_scores = self._clip_scores_batch(img_feat, attr_texts)
            attr_labels = [f"{obj}:{attr}" for obj, attr in g.attributes]
            result["attribute_scores"] = dict(zip(attr_labels, attr_scores))
            result["attribute_mean"]   = float(sum(attr_scores) / len(attr_scores))

        return result

    def truncate_prompt(self, text: str) -> str:
        """Truncate text to CLIP's 77-token limit."""
        tokens = self.tokenizer([text])[0]
        if tokens[-1].item() == 0:
            return text
        words = text.split()
        while len(words) > 1:
            words.pop()
            if self.tokenizer([" ".join(words)])[0][-1].item() == 0:
                return " ".join(words)
        return words[0] if words else text

    def sanity_check(self):
        """Quick smoke test with a blank image."""
        blank = Image.new("RGB", (64, 64))
        item  = PromptItem(text="a cat on a table", source="test")
        s     = self.compute(blank, item)
        print(f"Sanity check — blank image reward: {s.item():.4f}")
        return s


# ─────────────────────────────────────────────────────────────────────────────
# BLIP ITM — image-text matching reward (no extra pip installs needed)
# ─────────────────────────────────────────────────────────────────────────────

class BLIPRewardModel:
    """
    BLIP Image-Text Matching reward. Uses `transformers` (already installed).
    No new pip packages needed. No OpenAI `clip` dependency.

    BLIP-ITM outputs P(image matches text) ∈ [0, 1] via a binary classifier
    trained on image-text retrieval. A well-generated image matching its prompt
    scores 0.85-0.95; a mismatched or blank image scores 0.05-0.30.
    This gives a genuine ~0.9 spread vs CLIP ViT-B-32's ~0.15 spread —
    6× wider reward range for the same policy gradient update.

    baseline=0.5 centres the reward at zero (random image ≈ 0.5 ITM prob).
    scale=10.0 maps the ±0.5 range to ±5.0, matching the CLIP reward scale.

    One-time weight download on the login node (≈ 440 MB):
        HF_HOME=/scratch/3223837/.cache/huggingface \\
        python -c "from transformers import BlipProcessor, BlipForImageTextRetrieval; \\
                   BlipProcessor.from_pretrained('Salesforce/blip-itm-base-coco'); \\
                   BlipForImageTextRetrieval.from_pretrained('Salesforce/blip-itm-base-coco')"
    """

    def __init__(
        self,
        device:   str   = "cuda",
        baseline: float = 0.5,    # P(match) for a random/blank image
        scale:    float = 10.0,   # (p_match - 0.5) * 10 → reward in ~[-5, +5]
    ):
        from transformers import BlipProcessor, BlipForImageTextRetrieval
        print("Loading BLIP-ITM reward model (Salesforce/blip-itm-base-coco)...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        self.model     = BlipForImageTextRetrieval.from_pretrained(
            "Salesforce/blip-itm-base-coco",
            torch_dtype=torch.float32,
        ).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device   = device
        self.baseline = baseline
        self.scale    = scale
        print("BLIP-ITM reward model loaded.")

    @torch.no_grad()
    def compute(self, pil_image: Image.Image, prompt: str) -> torch.Tensor:
        """Return a scalar float32 reward on self.device."""
        inputs = self.processor(
            pil_image.convert("RGB"),
            prompt,
            return_tensors="pt",
        ).to(self.device)
        out     = self.model(**inputs, use_itm_head=True)
        # itm_score shape: [1, 2] logits — softmax → P(match) is index 1
        p_match = float(out.itm_score.softmax(dim=-1)[0, 1].cpu())
        scaled  = (p_match - self.baseline) * self.scale
        return torch.tensor(scaled, dtype=torch.float32, device=self.device)

    def sanity_check(self):
        blank = Image.new("RGB", (64, 64))
        s = self.compute(blank, "a cat on a table")
        print(f"BLIP-ITM sanity — blank image score: {s.item():.4f}  "
              f"(expected ≈ {(-self.baseline)*self.scale:.1f})")
        return s


# ─────────────────────────────────────────────────────────────────────────────
# COCO + PSG joint loader
# ─────────────────────────────────────────────────────────────────────────────

def load_coco_psg_items(
    coco_path: str,
    psg_path: str,
    max_items: int = 500,
    max_rel: int = 8,
    seed: int = 42,
) -> List[PromptItem]:
    """
    Join COCO captions with PSG scene graphs on image_id.

    For each matched image:
      - item.text  = one COCO caption  (used as the generation prompt)
      - item.graph = PSG SceneGraph    (used for hierarchical reward)

    Only images that appear in BOTH datasets are returned.
    """
    # Load PSG: build image_id → SceneGraph mapping
    with open(psg_path, encoding="utf-8") as f:
        psg_data = json.load(f)

    thing_classes = psg_data.get("thing_classes", [])
    stuff_classes = psg_data.get("stuff_classes", [])
    pred_classes  = psg_data.get("predicate_classes", [])
    obj_classes   = thing_classes + stuff_classes

    psg_by_id: Dict[int, SceneGraph] = {}
    for scene in psg_data.get("data", []):
        img_id   = scene.get("image_id")
        rels_raw = scene.get("relations", [])
        anns     = scene.get("annotations", [])
        if not img_id or not rels_raw or len(anns) < 2:
            continue

        obj_names = [_psg_obj_name(anns, obj_classes, i) for i in range(len(anns))]
        unique_objs = list(dict.fromkeys(obj_names))

        rel_triples: List[Tuple[str, str, str]] = []
        for rel in rels_raw[:max_rel]:
            s_idx, o_idx, p_idx = rel[0], rel[1], rel[2]
            sn = _psg_obj_name(anns, obj_classes, s_idx)
            on = _psg_obj_name(anns, obj_classes, o_idx)
            pn = pred_classes[p_idx] if 0 <= p_idx < len(pred_classes) else "near"
            rel_triples.append((sn, pn, on))

        psg_by_id[int(img_id)] = SceneGraph(
            objects=unique_objs[:10],
            attributes=[],
            relations=rel_triples,
        )

    print(f"[COCO+PSG] PSG scenes with valid graphs: {len(psg_by_id)}")

    # Load COCO captions: build image_id → [caption, ...] mapping
    with open(coco_path, encoding="utf-8") as f:
        coco_data = json.load(f)

    captions_by_id: Dict[int, List[str]] = {}
    for ann in coco_data.get("annotations", []):
        cap = ann.get("caption", "").strip()
        iid = ann.get("image_id")
        if cap and iid:
            captions_by_id.setdefault(int(iid), []).append(cap)

    # Join on image_id
    rng = random.Random(seed)
    matched_ids = list(set(psg_by_id.keys()) & set(captions_by_id.keys()))
    rng.shuffle(matched_ids)

    items = []
    for img_id in matched_ids:
        if len(items) >= max_items:
            break
        graph   = psg_by_id[img_id]
        caption = rng.choice(captions_by_id[img_id])

        # Build relation text for reference (not used as prompt)
        rel_phrases = [f"{s} {p} {o}" for s, p, o in graph.relations]
        relation_text = ("A scene where " + ", ".join(rel_phrases) + ".") if rel_phrases else caption

        items.append(PromptItem(
            text=caption,
            object_text="A photo containing " + ", ".join(graph.objects) + ".",
            relation_text=relation_text,
            graph=graph,
            source="coco_psg",
            image_id=str(img_id),
        ))

    print(f"[COCO+PSG] {len(items)} matched items (from {len(matched_ids)} overlapping image IDs)")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical reward model (COCO prompt + PSG graph)
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalRewardModel:
    """
    Hierarchical CLIP reward that decomposes scene graph structure.

    reward = w_global * CLIP(image, full_caption)
           + w_obj    * mean(CLIP(image, "a photo of {obj}") for obj in graph.objects)
           + w_rel    * mean(CLIP(image, "{s} {p} {o}") for s, p, o in graph.relations)
           + w_attr   * mean(CLIP(image, "a {attr} {obj}") for obj, attr in graph.attributes)

    Then reward = (combined - baseline) * scale

    Falls back to pure global CLIPScore if item.graph is None.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        # Component weights (must roughly sum to 1.0 for a consistent scale)
        w_global: float = 0.30,
        w_obj:    float = 0.35,
        w_rel:    float = 0.25,
        w_attr:   float = 0.10,
        baseline: float = 0.22,
        scale:    float = 10.0,
    ):
        self.device   = device
        self.w_global = w_global
        self.w_obj    = w_obj
        self.w_rel    = w_rel
        self.w_attr   = w_attr
        self.baseline = baseline
        self.scale    = scale

        print(f"Loading hierarchical CLIP reward model ({model_name})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        self.model     = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        print("Hierarchical reward model loaded.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        img_t = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return F.normalize(self.model.encode_image(img_t), dim=-1)

    def _clip_score(self, img_feat: torch.Tensor, text: str) -> float:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            txt_feat = F.normalize(self.model.encode_text(tokens), dim=-1)
        return float((img_feat * txt_feat).sum(dim=-1).cpu())

    def _clip_batch(self, img_feat: torch.Tensor, texts: List[str]) -> List[float]:
        if not texts:
            return []
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            txt_feats = F.normalize(self.model.encode_text(tokens), dim=-1)
        return (img_feat @ txt_feats.T).squeeze(0).cpu().tolist()

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_components(self, pil_image: Image.Image, item: PromptItem) -> dict:
        """
        Returns all component scores as a dict.
        Keys: global, object_mean, relation_mean, attr_mean, combined, reward
        """
        img_feat = self._encode_image(pil_image)
        g = item.graph

        # Global (always computed)
        global_score = self._clip_score(img_feat, item.text)

        components = {"global": global_score}

        if g is None or len(g.objects) < 1:
            # No graph: full weight on global
            combined = global_score
            components["object_mean"]   = None
            components["relation_mean"] = None
            components["attr_mean"]     = None
        else:
            # Object recall
            obj_texts  = [f"a photo of a {o}" for o in g.objects]
            obj_scores = self._clip_batch(img_feat, obj_texts)
            obj_mean   = float(sum(obj_scores) / len(obj_scores)) if obj_scores else global_score
            components["object_scores"] = dict(zip(g.objects, obj_scores))
            components["object_mean"]   = obj_mean

            # Relation score
            rel_mean = None
            if g.relations:
                rel_texts  = [f"{s} {p} {o}" for s, p, o in g.relations]
                rel_scores = self._clip_batch(img_feat, rel_texts)
                rel_mean   = float(sum(rel_scores) / len(rel_scores))
                components["relation_scores"] = dict(
                    zip([f"{s}-{p}-{o}" for s, p, o in g.relations], rel_scores)
                )
            components["relation_mean"] = rel_mean

            # Attribute score
            attr_mean = None
            if g.attributes:
                attr_texts  = [f"a {attr} {obj}" for obj, attr in g.attributes]
                attr_scores = self._clip_batch(img_feat, attr_texts)
                attr_mean   = float(sum(attr_scores) / len(attr_scores))
                components["attribute_scores"] = dict(
                    zip([f"{obj}:{attr}" for obj, attr in g.attributes], attr_scores)
                )
            components["attr_mean"] = attr_mean

            # Weighted combination — redistribute weights if components are absent
            w_g, w_o, w_r, w_a = self.w_global, self.w_obj, self.w_rel, self.w_attr
            if rel_mean is None:
                extra = w_r
                w_r = 0.0
                w_g += extra * (w_g / (w_g + w_o + w_a + 1e-9))
                w_o += extra * (w_o / (w_g + w_o + w_a + 1e-9))
            if attr_mean is None:
                extra = w_a
                w_a = 0.0
                w_g += extra * (w_g / (w_g + w_o + w_r + 1e-9))
                w_o += extra * (w_o / (w_g + w_o + w_r + 1e-9))

            combined = (
                w_g * global_score
                + w_o * obj_mean
                + (w_r * rel_mean if rel_mean is not None else 0.0)
                + (w_a * attr_mean if attr_mean is not None else 0.0)
            )

        reward_val = (combined - self.baseline) * self.scale
        components["combined"] = combined
        components["reward"]   = reward_val
        return components

    @torch.no_grad()
    def compute(self, pil_image: Image.Image, item: PromptItem) -> torch.Tensor:
        """Returns scalar float32 reward tensor."""
        c = self.compute_components(pil_image, item)
        return torch.tensor(c["reward"], dtype=torch.float32, device=self.device)

    def sanity_check(self):
        blank = Image.new("RGB", (64, 64))
        graph = SceneGraph(objects=["cat", "table"], attributes=[], relations=[("cat", "on", "table")])
        item  = PromptItem(text="a cat on a table", graph=graph, source="test")
        c     = self.compute_components(blank, item)
        print(f"Hierarchical sanity check — reward: {c['reward']:.4f}  "
              f"global: {c['global']:.4f}  obj: {c['object_mean']:.4f}  "
              f"rel: {c['relation_mean']:.4f}")
