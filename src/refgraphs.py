from rewards.rewards import Node, Binding, Interaction, Graph


PRONOUN = {
    "man": "his",
    "woman": "her",
    "boy": "his",
    "girl": "her",
}

SUBJECTS = ("man", "woman", "boy", "girl")
ANIMALS = ("cat", "dog", "rabbit")
TSHIRT_COLORS = ("white", "green", "black")
PANTS_STYLES = ("blue jeans", "cargo pants", "ripped skinny jeans")
SHOES_STYLES = ("bright red sneakers", "brown leather boots", "white running shoes")
GLASSES_STYLE = "rectangular glasses"
GLASSES_NEGATIVES = ("round glasses", "sunglasses", "without glasses")
ANIMAL_INTERACTIONS = ("holding", "walking")


def _exclude(options, selected):
    return [value for value in options if value != selected]


def _unique(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _subject_glasses_prompt(subject, glasses_style):
    if glasses_style == "without glasses":
        return f"a {subject} without glasses"
    return f"a {subject} wearing {glasses_style}"


def _holding_prompt(subject, animal):
    possessive = PRONOUN.get(subject, "their")
    return f"a {subject} gently holding a fluffy {animal} in {possessive} arms"


def _walking_prompt(subject, animal):
    return f"a {subject} walking a {animal} on a leash"


def _animal_interaction_prompt(subject, animal, interaction_mode):
    if interaction_mode == "holding":
        return _holding_prompt(subject, animal)
    if interaction_mode == "walking":
        return _walking_prompt(subject, animal)
    raise ValueError(f"Unsupported interaction mode: {interaction_mode}")


def _overall_prompt_from_interactions(interactions):
    return ", ".join(item.prompt for item in interactions)


def _clip_safe_overall_prompt(
    subject,
    animal,
    tshirt_color,
    pants_style,
    shoes_style,
    animal_interaction,
):
    if animal_interaction == "holding":
        animal_action = f"gently holding a fluffy {animal}"
    elif animal_interaction == "walking":
        animal_action = f"walking a {animal} on a leash"
    else:
        animal_action = f"{animal_interaction} a {animal}"

    return (
        f"a casually dressed {subject} wearing {GLASSES_STYLE}, "
        f"a {tshirt_color} t-shirt, {pants_style}, and {shoes_style}; "
        f"{animal_action}"
    )


def build_graph(
    subject="man",
    animal="cat",
    tshirt_color="white",
    pants_style="blue jeans",
    shoes_style="bright red sneakers",
    animal_interaction="holding",
    return_overall_prompt=False,
    overall_prompt_mode="clip_safe",
):
    # ---- Nodes ----
    nodes = [
        Node(
            subject,
            f"a casually dressed {subject}",
            neg_prompts=[f"a casually dressed {s}" for s in _exclude(SUBJECTS, subject)],
        ),
        Node("glasses", GLASSES_STYLE, neg_prompts=list(GLASSES_NEGATIVES)),
        Node(
            "tshirt",
            f"a {tshirt_color} t-shirt",
            neg_prompts=[f"a {c} t-shirt" for c in _exclude(TSHIRT_COLORS, tshirt_color)],
        ),
        Node("pants", pants_style, neg_prompts=_exclude(PANTS_STYLES, pants_style)),
        Node("shoes", shoes_style, neg_prompts=_exclude(SHOES_STYLES, shoes_style)),
        Node(
            "animal",
            f"a {animal}",
            neg_prompts=[f"a {a}" for a in _exclude(ANIMALS, animal)],
        ),
    ]

    # ---- Bindings ----
    bindings = [
        Binding("glasses", GLASSES_STYLE, neg_prompts=list(GLASSES_NEGATIVES)),
        Binding(
            "tshirt",
            f"a {tshirt_color} t-shirt",
            neg_prompts=[f"a {c} t-shirt" for c in _exclude(TSHIRT_COLORS, tshirt_color)],
        ),
        Binding("pants", pants_style, neg_prompts=_exclude(PANTS_STYLES, pants_style)),
        Binding("shoes", shoes_style, neg_prompts=_exclude(SHOES_STYLES, shoes_style)),
        Binding(
            "animal",
            f"a fluffy {animal}",
            neg_prompts=[f"a fluffy {a}" for a in _exclude(ANIMALS, animal)],
        ),
    ]

    # ---- Interactions ----
    interactions = [
        Interaction(
            subject,
            "glasses",
            _subject_glasses_prompt(subject, GLASSES_STYLE),
            neg_prompts=_unique(
                [_subject_glasses_prompt(s, GLASSES_STYLE) for s in _exclude(SUBJECTS, subject)]
                + [_subject_glasses_prompt(subject, g) for g in GLASSES_NEGATIVES]
            ),
        ),
        Interaction(
            subject,
            "tshirt",
            f"a {subject} wearing a {tshirt_color} t-shirt",
            neg_prompts=_unique(
                [f"a {subject} wearing a {c} t-shirt" for c in _exclude(TSHIRT_COLORS, tshirt_color)]
                + [f"a {s} wearing a {tshirt_color} t-shirt" for s in _exclude(SUBJECTS, subject)]
            ),
        ),
        Interaction(
            subject,
            "pants",
            f"a {subject} wearing {pants_style}",
            neg_prompts=_unique(
                [f"a {subject} wearing {p}" for p in _exclude(PANTS_STYLES, pants_style)]
                + [f"a {s} wearing {pants_style}" for s in _exclude(SUBJECTS, subject)]
            ),
        ),
        Interaction(
            subject,
            "shoes",
            f"a {subject} wearing {shoes_style}",
            neg_prompts=_unique(
                [f"a {subject} wearing {s}" for s in _exclude(SHOES_STYLES, shoes_style)]
                + [f"a {s} wearing {shoes_style}" for s in _exclude(SUBJECTS, subject)]
            ),
        ),
        Interaction(
            subject,
            "animal",
            _animal_interaction_prompt(subject, animal, animal_interaction),
            neg_prompts=_unique(
                [
                    _animal_interaction_prompt(subject, a, animal_interaction)
                    for a in _exclude(ANIMALS, animal)
                ]
                + [
                    _animal_interaction_prompt(s, animal, animal_interaction)
                    for s in _exclude(SUBJECTS, subject)
                ]
                + [
                    _animal_interaction_prompt(subject, animal, mode)
                    for mode in _exclude(ANIMAL_INTERACTIONS, animal_interaction)
                ]
            ),
        ),
    ]

    graph = Graph(nodes, bindings, interactions)
    if return_overall_prompt:
        if overall_prompt_mode == "verbose":
            overall_prompt = _overall_prompt_from_interactions(interactions)
        elif overall_prompt_mode == "clip_safe":
            overall_prompt = _clip_safe_overall_prompt(
                subject=subject,
                animal=animal,
                tshirt_color=tshirt_color,
                pants_style=pants_style,
                shoes_style=shoes_style,
                animal_interaction=animal_interaction,
            )
        else:
            raise ValueError(
                f"Unsupported overall_prompt_mode: {overall_prompt_mode}. "
                "Expected one of: clip_safe, verbose."
            )
        return graph, overall_prompt
    return graph

# The reference EXACT graph and 9 alternatives with various modifications, used in src/simple_hier_clip_reward.py for evaluation.
# The alternatives include changes to the subject, animal, clothing attributes, and interactions, designed to test the sensitivity of the reward model to different types of modifications.
# Original prompt for EXACT graph: "a casually dressed man wearing rectangular glasses, a white t-shirt, blue jeans, bright red sneakers, gently holding a fluffy cat in his arms."

EXACT = build_graph()
ALT1 = build_graph(subject="woman")
ALT2 = build_graph(animal="dog")
ALT3 = build_graph(animal="rabbit")
ALT4 = build_graph(tshirt_color="green")
ALT5 = build_graph(pants_style="cargo pants")
ALT6 = build_graph(shoes_style="brown leather boots")
ALT7 = build_graph(
    subject="woman",
    animal="dog",
    tshirt_color="black",
    pants_style="ripped skinny jeans",
    shoes_style="white running shoes",
)
ALT8 = build_graph(
    subject="man",
    animal="rabbit",
    tshirt_color="green",
    pants_style="cargo pants",
    shoes_style="brown leather boots",
)
ALT9 = build_graph(
    subject="man",
    animal="dog",
    animal_interaction="walking",
)
