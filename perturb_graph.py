"""
Scene-graph perturbations for the sensitivity test harness.

Each perturbation takes a SceneGraph and returns a *new* SceneGraph
plus a short string label describing what changed (useful for table
captions and CSV columns).

No neural models, no image touching — graph-only string ops.

Functions:
    perturb_remove_object(graph)        — drop one object + its deps
    perturb_swap_attribute(graph)       — flip color/size to its antonym
    perturb_swap_spatial_relation(graph)— flip left↔right etc.
    perturb_swap_unrelated_noun(graph)  — replace an object with a distractor
"""

from __future__ import annotations

import copy
import random
from typing import Optional, Tuple

from dpok_scene_reward import SceneGraph


# ─────────────────────────────────────────────────────────────────────────────
# Antonym tables — hand-curated, the part that matters most for sensitivity
# ─────────────────────────────────────────────────────────────────────────────

ATTRIBUTE_ANTONYMS = {
    # colors
    "red": "green", "green": "red",
    "blue": "orange", "orange": "blue",
    "yellow": "purple", "purple": "yellow",
    "black": "white", "white": "black",
    "pink": "brown", "brown": "pink",
    "gray": "vibrant", "grey": "vibrant",
    # patterns / appearance (from FIBO examples)
    "striped": "solid-colored", "solid-colored": "striped",
    "pastel": "neon", "neon": "pastel",
    # sizes
    "big": "small", "small": "big",
    "large": "tiny", "tiny": "large",
    "tall": "short", "short": "tall",
    "long": "short", "wide": "narrow", "narrow": "wide",
    # textures / qualities
    "rough": "smooth", "smooth": "rough",
    "shiny": "dull", "dull": "shiny",
    "bright": "dim", "dim": "bright",
    "old": "new", "new": "old",
    "young": "old",
    "wet": "dry", "dry": "wet",
    "full": "empty", "empty": "full",
    "open": "closed", "closed": "open",
    "hot": "cold", "cold": "hot",
    # orientations
    "left-facing": "right-facing", "right-facing": "left-facing",
}

SPATIAL_RELATIONS = {
    "left of": "right of", "right of": "left of",
    "above": "below", "below": "above",
    "on top of": "underneath", "underneath": "on top of",
    "in front of": "behind", "behind": "in front of",
    "inside": "outside", "outside": "inside",
    "near": "far from", "far from": "near",
    "next to": "far from",
}

DISTRACTOR_NOUNS = [
    "spaceship", "violin", "cactus", "lighthouse", "telescope",
    "octopus", "chandelier", "barbell", "tuba", "iceberg",
]


# ─────────────────────────────────────────────────────────────────────────────
# Perturbations
# ─────────────────────────────────────────────────────────────────────────────

def _clone(g: SceneGraph) -> SceneGraph:
    return SceneGraph(
        objects=list(g.objects),
        attributes=list(g.attributes),
        relations=list(g.relations),
    )


def perturb_remove_object(
    graph: SceneGraph, rng: Optional[random.Random] = None
) -> Tuple[SceneGraph, str]:
    """Drop one object and every attribute/relation that mentions it.

    Returns (perturbed_graph, label). label is the removed object name.
    Returns the input graph unchanged if there are no objects.
    """
    rng = rng or random.Random()
    if not graph.objects:
        return _clone(graph), "no-op"
    victim = rng.choice(graph.objects)
    new = SceneGraph(
        objects=[o for o in graph.objects if o != victim],
        attributes=[(o, a) for (o, a) in graph.attributes if o != victim],
        relations=[(s, r, o) for (s, r, o) in graph.relations
                   if s != victim and o != victim],
    )
    return new, f"removed:{victim}"


def perturb_swap_attribute(
    graph: SceneGraph, rng: Optional[random.Random] = None
) -> Tuple[SceneGraph, str]:
    """Replace one attribute with its antonym. Skips unknown attributes."""
    rng = rng or random.Random()
    swappable = [
        i for i, (_, a) in enumerate(graph.attributes)
        if a.lower() in ATTRIBUTE_ANTONYMS
    ]
    if not swappable:
        return _clone(graph), "no-swappable-attribute"
    idx = rng.choice(swappable)
    obj, attr = graph.attributes[idx]
    new_attr = ATTRIBUTE_ANTONYMS[attr.lower()]
    new_attrs = list(graph.attributes)
    new_attrs[idx] = (obj, new_attr)
    new = SceneGraph(
        objects=list(graph.objects),
        attributes=new_attrs,
        relations=list(graph.relations),
    )
    return new, f"{obj}:{attr}->{new_attr}"


def perturb_swap_spatial_relation(
    graph: SceneGraph, rng: Optional[random.Random] = None
) -> Tuple[SceneGraph, str]:
    """Flip one spatial relation (left↔right, above↔below, ...)."""
    rng = rng or random.Random()
    swappable = [
        i for i, (_, r, _) in enumerate(graph.relations)
        if r.lower() in SPATIAL_RELATIONS
    ]
    if not swappable:
        return _clone(graph), "no-spatial-relation"
    idx = rng.choice(swappable)
    s, r, o = graph.relations[idx]
    new_r = SPATIAL_RELATIONS[r.lower()]
    new_rels = list(graph.relations)
    new_rels[idx] = (s, new_r, o)
    new = SceneGraph(
        objects=list(graph.objects),
        attributes=list(graph.attributes),
        relations=new_rels,
    )
    return new, f"{s} {r}->{new_r} {o}"


def perturb_swap_unrelated_noun(
    graph: SceneGraph, rng: Optional[random.Random] = None
) -> Tuple[SceneGraph, str]:
    """Sanity floor: replace an object with an unrelated distractor.
    All rewards should respond moderately — this is not a compositional test."""
    rng = rng or random.Random()
    if not graph.objects:
        return _clone(graph), "no-op"
    victim = rng.choice(graph.objects)
    distractors = [d for d in DISTRACTOR_NOUNS if d not in graph.objects]
    if not distractors:
        return _clone(graph), "no-distractor"
    replacement = rng.choice(distractors)
    new = SceneGraph(
        objects=[replacement if o == victim else o for o in graph.objects],
        attributes=[(replacement if o == victim else o, a)
                    for (o, a) in graph.attributes],
        relations=[
            (replacement if s == victim else s, r,
             replacement if o == victim else o)
            for (s, r, o) in graph.relations
        ],
    )
    return new, f"{victim}->{replacement}"


# ─────────────────────────────────────────────────────────────────────────────
# Convenience
# ─────────────────────────────────────────────────────────────────────────────

PERTURBATIONS = {
    "remove_object":   perturb_remove_object,
    "swap_attribute":  perturb_swap_attribute,
    "swap_spatial":    perturb_swap_spatial_relation,
    "swap_noun":       perturb_swap_unrelated_noun,
}


def stringify_graph(g: SceneGraph) -> str:
    """Render a SceneGraph as a natural-ish caption for the baseline scorers.

    "a red rabbit, a green leaf; rabbit on leaf"
    """
    parts = []
    attr_map = {}
    for obj, attr in g.attributes:
        attr_map.setdefault(obj, []).append(attr)
    for obj in g.objects:
        adjs = " ".join(attr_map.get(obj, []))
        parts.append(f"a {adjs} {obj}".strip().replace("  ", " "))
    obj_str = ", ".join(parts) if parts else ""
    rel_str = "; ".join(f"{s} {r} {o}" for s, r, o in g.relations)
    if obj_str and rel_str:
        return f"{obj_str}; {rel_str}"
    return obj_str or rel_str or "an image"
