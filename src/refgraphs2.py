from rewards.rewards import Node, Binding, Interaction, Graph


PRONOUN = {
    "woman": "her",
    "man": "his",
    "girl": "her",
    "boy": "his",
}

SUBJECTS = ("woman", "man", "girl", "boy")
HAIR_STYLES = (
    "shoulder-length dark hair",
    "long blonde hair",
    "short curly red hair",
)
GLASSES_STYLES = ("thin round glasses", "rectangular glasses", "without glasses")
SHIRT_STYLES = (
    "a fitted black t-shirt",
    "a fitted white t-shirt",
    "a loose green t-shirt",
)
JEANS_STYLES = (
    "high-waisted light blue jeans",
    "dark blue jeans",
    "black cargo pants",
)
SHOES_STYLES = ("white running shoes", "red sneakers", "brown leather boots")
DOG_DESCRIPTIONS = (
    "a small brown terrier mix with floppy ears, dark round eyes, a black nose, and a slightly curled tail",
    "a black labrador dog with a short coat",
    "a fluffy white poodle with curly fur",
)
DOG_NODE_NEGATIVES = ("a cat", "a rabbit", "a stuffed toy dog")
POSES = ("standing upright", "sitting on a bench")
SCENES = (
    "a grassy park lawn with trimmed green grass and scattered trees",
    "a city sidewalk with buildings and traffic",
    "an indoor studio backdrop",
)
LIGHTING_OPTIONS = ("soft natural daylight", "overcast daylight", "nighttime street lighting")
DOG_INTERACTIONS = ("holding", "walking")
DOG_INTERACTION_MODE_TEXT = {
    "holding": "the dog is being held in arms",
    "walking": "the dog is being walked on a leash",
}
DOG_POSTURES = (
    "resting calmly against her chest with its front paws draped over her forearm",
    "standing on the ground beside her",
    "running excitedly ahead",
)
SCENE_CONTEXT_MAP = {
    "a grassy park lawn with trimmed green grass and scattered trees": "an outdoor park setting",
    "a city sidewalk with buildings and traffic": "an urban sidewalk setting",
    "an indoor studio backdrop": "an indoor studio setting",
}
SCENE_CONTEXTS = tuple(SCENE_CONTEXT_MAP.values())


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


def _glasses_prompt(subject, glasses_style):
    if glasses_style == "without glasses":
        return f"a {subject} without glasses"
    return f"a {subject} wearing {glasses_style}"


def _dog_interaction_prompt(subject, dog_desc, interaction_mode):
    possessive = PRONOUN.get(subject, "their")
    if interaction_mode == "holding":
        return f"a {subject} gently holding {dog_desc} in {possessive} arms"
    if interaction_mode == "walking":
        return f"a {subject} walking {dog_desc} on a leash"
    raise ValueError(f"Unsupported interaction mode: {interaction_mode}")


def _dog_mode_prompt(subject, interaction_mode):
    possessive = PRONOUN.get(subject, "their")
    if interaction_mode == "holding":
        return f"a {subject} gently holding the dog in {possessive} arms"
    if interaction_mode == "walking":
        return f"a {subject} walking the dog on a leash"
    raise ValueError(f"Unsupported interaction mode: {interaction_mode}")


def _dog_posture_prompt(subject, dog_posture):
    possessive = PRONOUN.get(subject, "their")
    return f"the dog {dog_posture.replace('her', possessive)}"


def _overall_prompt_from_interactions(interactions):
    return ", ".join(item.prompt for item in interactions)


def _clip_safe_overall_prompt(
    subject,
    hairstyle,
    glasses_style,
    shirt_style,
    jeans_style,
    shoes_style,
    dog_desc,
    pose,
    scene_context,
    lighting,
    dog_interaction,
    dog_posture,
):
    dog_short = dog_desc.split(",", 1)[0].strip()

    if glasses_style == "without glasses":
        glasses_phrase = "without glasses"
    else:
        glasses_phrase = f"wearing {glasses_style}"

    if dog_interaction == "holding":
        dog_action = f"gently holding {dog_short}"
    elif dog_interaction == "walking":
        dog_action = f"walking {dog_short} on a leash"
    else:
        dog_action = f"{dog_interaction} {dog_short}"

    dog_posture_phrase = _dog_posture_prompt(subject, dog_posture)
    if dog_posture_phrase.startswith("the dog "):
        dog_posture_phrase = dog_posture_phrase[len("the dog "):]
    dog_posture_short = dog_posture_phrase.split(" with ", 1)[0].strip()

    return (
        f"a casually dressed {subject} with {hairstyle}, {glasses_phrase}, "
        f"{shirt_style}, {jeans_style}, and {shoes_style}; "
        f"{dog_action}; dog {dog_posture_short}; "
        f"{pose} in {scene_context}, {lighting}"
    )


def build_graph2(
    subject="woman",
    hairstyle="shoulder-length dark hair",
    glasses_style="thin round glasses",
    shirt_style="a fitted black t-shirt",
    jeans_style="high-waisted light blue jeans",
    shoes_style="white running shoes",
    dog_desc="a small brown terrier mix with floppy ears, dark round eyes, a black nose, and a slightly curled tail",
    pose="standing upright",
    scene="a grassy park lawn with trimmed green grass and scattered trees",
    lighting="soft natural daylight",
    dog_interaction="holding",
    dog_posture="resting calmly against her chest with its front paws draped over her forearm",
    return_overall_prompt=False,
    overall_prompt_mode="clip_safe",
):
    scene_context = SCENE_CONTEXT_MAP[scene]
    dog_interaction_mode_text = DOG_INTERACTION_MODE_TEXT[dog_interaction]

    nodes = [
        Node(
            subject,
            f"a casually dressed {subject}",
            neg_prompts=[f"a casually dressed {s}" for s in _exclude(SUBJECTS, subject)],
        ),
        Node("hair", hairstyle, neg_prompts=_exclude(HAIR_STYLES, hairstyle)),
        Node("glasses", glasses_style, neg_prompts=_exclude(GLASSES_STYLES, glasses_style)),
        Node("shirt", shirt_style, neg_prompts=_exclude(SHIRT_STYLES, shirt_style)),
        Node("jeans", jeans_style, neg_prompts=_exclude(JEANS_STYLES, jeans_style)),
        Node("shoes", shoes_style, neg_prompts=_exclude(SHOES_STYLES, shoes_style)),
        Node("dog", "a dog", neg_prompts=list(DOG_NODE_NEGATIVES)),
        Node(
            "dog_interaction_mode",
            dog_interaction_mode_text,
            neg_prompts=[DOG_INTERACTION_MODE_TEXT[m] for m in _exclude(DOG_INTERACTIONS, dog_interaction)],
        ),
        Node("pose", pose, neg_prompts=_exclude(POSES, pose)),
        Node("scene", scene, neg_prompts=_exclude(SCENES, scene)),
        Node("scene_context", scene_context, neg_prompts=_exclude(SCENE_CONTEXTS, scene_context)),
        Node("lighting", lighting, neg_prompts=_exclude(LIGHTING_OPTIONS, lighting)),
    ]

    bindings = [
        Binding("hair", hairstyle, neg_prompts=_exclude(HAIR_STYLES, hairstyle)),
        Binding("glasses", glasses_style, neg_prompts=_exclude(GLASSES_STYLES, glasses_style)),
        Binding("shirt", shirt_style, neg_prompts=_exclude(SHIRT_STYLES, shirt_style)),
        Binding("jeans", jeans_style, neg_prompts=_exclude(JEANS_STYLES, jeans_style)),
        Binding("shoes", shoes_style, neg_prompts=_exclude(SHOES_STYLES, shoes_style)),
        Binding("dog", dog_desc, neg_prompts=_exclude(DOG_DESCRIPTIONS, dog_desc)),
        Binding(
            "dog_interaction_mode",
            dog_interaction_mode_text,
            neg_prompts=[DOG_INTERACTION_MODE_TEXT[m] for m in _exclude(DOG_INTERACTIONS, dog_interaction)],
        ),
        Binding("dog", _dog_posture_prompt(subject, dog_posture), neg_prompts=[
            _dog_posture_prompt(subject, p) for p in _exclude(DOG_POSTURES, dog_posture)
        ]),
        Binding("scene", scene, neg_prompts=_exclude(SCENES, scene)),
        Binding("scene_context", scene_context, neg_prompts=_exclude(SCENE_CONTEXTS, scene_context)),
        Binding("lighting", lighting, neg_prompts=_exclude(LIGHTING_OPTIONS, lighting)),
    ]

    interactions = [
        Interaction(
            subject,
            "hair",
            f"a {subject} with {hairstyle}",
            neg_prompts=_unique(
                [f"a {subject} with {h}" for h in _exclude(HAIR_STYLES, hairstyle)]
                + [f"a {s} with {hairstyle}" for s in _exclude(SUBJECTS, subject)]
            ),
        ),
        Interaction(
            subject,
            "glasses",
            _glasses_prompt(subject, glasses_style),
            neg_prompts=_unique(
                [_glasses_prompt(subject, g) for g in _exclude(GLASSES_STYLES, glasses_style)]
                + [_glasses_prompt(s, glasses_style) for s in _exclude(SUBJECTS, subject)]
            ),
        ),
        Interaction(
            subject,
            "shirt",
            f"a {subject} wearing {shirt_style}",
            neg_prompts=_unique(
                [f"a {subject} wearing {s}" for s in _exclude(SHIRT_STYLES, shirt_style)]
                + [f"a {s} wearing {shirt_style}" for s in _exclude(SUBJECTS, subject)]
            ),
        ),
        Interaction(
            subject,
            "jeans",
            f"a {subject} wearing {jeans_style}",
            neg_prompts=_unique(
                [f"a {subject} wearing {j}" for j in _exclude(JEANS_STYLES, jeans_style)]
                + [f"a {s} wearing {jeans_style}" for s in _exclude(SUBJECTS, subject)]
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
            "dog",
            _dog_interaction_prompt(subject, dog_desc, dog_interaction),
            neg_prompts=_unique(
                [
                    _dog_interaction_prompt(subject, d, dog_interaction)
                    for d in _exclude(DOG_DESCRIPTIONS, dog_desc)
                ]
                + [
                    _dog_interaction_prompt(s, dog_desc, dog_interaction)
                    for s in _exclude(SUBJECTS, subject)
                ]
                + [
                    _dog_interaction_prompt(subject, dog_desc, mode)
                    for mode in _exclude(DOG_INTERACTIONS, dog_interaction)
                ]
            ),
        ),
        Interaction(
            subject,
            "dog_interaction_mode",
            _dog_mode_prompt(subject, dog_interaction),
            neg_prompts=_unique(
                [_dog_mode_prompt(subject, m) for m in _exclude(DOG_INTERACTIONS, dog_interaction)]
                + [_dog_mode_prompt(s, dog_interaction) for s in _exclude(SUBJECTS, subject)]
            ),
        ),
        Interaction(
            "dog",
            subject,
            _dog_posture_prompt(subject, dog_posture),
            neg_prompts=[_dog_posture_prompt(subject, p) for p in _exclude(DOG_POSTURES, dog_posture)],
        ),
        Interaction(
            subject,
            "scene",
            f"a {subject} {pose} on {scene}",
            neg_prompts=_unique(
                [f"a {subject} {pose} on {s}" for s in _exclude(SCENES, scene)]
                + [f"a {subject} {p} on {scene}" for p in _exclude(POSES, pose)]
            ),
        ),
        Interaction(
            subject,
            "scene_context",
            f"a {subject} in {scene_context}",
            neg_prompts=_unique(
                [f"a {subject} in {c}" for c in _exclude(SCENE_CONTEXTS, scene_context)]
                + [f"a {s} in {scene_context}" for s in _exclude(SUBJECTS, subject)]
            ),
        ),
        Interaction(
            subject,
            "lighting",
            f"{lighting} illuminating both the {subject} and the dog",
            neg_prompts=[
                f"{l} illuminating both the {subject} and the dog"
                for l in _exclude(LIGHTING_OPTIONS, lighting)
            ],
        ),
    ]

    graph = Graph(nodes, bindings, interactions)
    if return_overall_prompt:
        if overall_prompt_mode == "verbose":
            overall_prompt = _overall_prompt_from_interactions(interactions)
        elif overall_prompt_mode == "clip_safe":
            overall_prompt = _clip_safe_overall_prompt(
                subject=subject,
                hairstyle=hairstyle,
                glasses_style=glasses_style,
                shirt_style=shirt_style,
                jeans_style=jeans_style,
                shoes_style=shoes_style,
                dog_desc=dog_desc,
                pose=pose,
                scene_context=scene_context,
                lighting=lighting,
                dog_interaction=dog_interaction,
                dog_posture=dog_posture,
            )
        else:
            raise ValueError(
                f"Unsupported overall_prompt_mode: {overall_prompt_mode}. "
                "Expected one of: clip_safe, verbose."
            )
        return graph, overall_prompt
    return graph


EXACT2 = build_graph2()
ALT2_1 = build_graph2(subject="man")
ALT2_2 = build_graph2(glasses_style="rectangular glasses")
ALT2_3 = build_graph2(shirt_style="a fitted white t-shirt")
ALT2_4 = build_graph2(dog_interaction="walking")
ALT2_5 = build_graph2(scene="a city sidewalk with buildings and traffic")
ALT2_6 = build_graph2(lighting="nighttime street lighting")
