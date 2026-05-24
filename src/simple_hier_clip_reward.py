import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

from rewards.clip import CLIPScorer
from rewards.rewards import SimpleHierReward
from viz.viz_graph import render_reward_graph_weighted
from refgraphs import build_graph
from refgraphs2 import build_graph2


def load_image_tensor(image_path, device):
    with Image.open(image_path) as img:
        return transforms.ToTensor()(img.convert("RGB")).unsqueeze(0).to(device).requires_grad_(True)


def resolve_image_path(data_dir, preferred_names):
    for name in preferred_names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"None of the expected images were found in {data_dir}: {preferred_names}"
    )


def compute_plain_clip_score(scorer, image_feat, full_prompt):
    clip_score = scorer.score_from_image_feature(image_feat, full_prompt)
    return float(clip_score.detach().cpu())


def format_graph_summary(graph_name, graph, full_prompt):
    node_names = ", ".join(n.name for n in graph.nodes)
    interaction_lines = "\n".join(
        f"    - {it.src} -> {it.dst}: {it.prompt}" for it in graph.interactions
    )
    return (
        f"\n=== Evaluating {graph_name} ===\n"
        f"  Full prompt: {full_prompt}\n"
        f"  Nodes ({len(graph.nodes)}): {node_names}\n"
        f"  Bindings: {len(graph.bindings)}\n"
        f"  Interactions: {len(graph.interactions)}\n"
        f"{interaction_lines}"
    )


def evaluate_graph_set(set_name, graphs, scorer, image, image_feat, weights):
    baseline_entries = [entry for entry in graphs if entry[0] in ("EXACT", "EXACT2")]
    if not baseline_entries:
        raise ValueError(
            f"No EXACT baseline found for set '{set_name}'. "
            "Expected a graph name like EXACT or EXACT2."
        )
    baseline_name, baseline_graph, baseline_prompt = baseline_entries[0]

    baseline_model = SimpleHierReward(
        scorer,
        baseline_graph,
        weights=weights,
        gating="soft",
        alpha=10.0,
        gate_tau=0.5,
        contrastive_temp=0.03,
        contrastive_mode="prob",
    ).to(image.device)
    baseline_R, _ = baseline_model(image)
    baseline_hier_reward = float(baseline_R.detach().cpu())
    baseline_clip_score = compute_plain_clip_score(scorer, image_feat, baseline_prompt)

    print(f"\n===== {set_name.upper()} =====")
    for graph_name, graph, full_prompt in graphs:
        print(format_graph_summary(graph_name, graph, full_prompt))
        model = SimpleHierReward(
            scorer, graph,
            weights=weights,
            gating="soft", alpha=10.0, gate_tau=0.5, contrastive_temp=0.03, contrastive_mode="prob",
        ).to(image.device)

        R, dbg = model(image)
        reward = float(R.detach().cpu())
        clip_score = compute_plain_clip_score(scorer, image_feat, full_prompt)

        delta_hier = reward - baseline_hier_reward
        delta_clip = clip_score - baseline_clip_score

        render_reward_graph_weighted(
            graph.nodes,
            graph.bindings,
            graph.interactions,
            dbg,
            weights=weights,
            out_path=f"reward_graph_weighted_{set_name}_{graph_name}"
        )
        print(f"Hier reward: {reward:+.4f} (delta vs {baseline_name}: {delta_hier:+.4f})")
        print(f"Plain CLIP score: {clip_score:+.4f} (delta vs {baseline_name}: {delta_clip:+.4f})")
        if graph_name == baseline_name:
            print("Sensitivity gain vs CLIP: baseline")
        else:
            extra_drop = abs(delta_hier) - abs(delta_clip)
            if abs(delta_clip) > 1e-8:
                relative_gain_pct = (abs(delta_hier) / abs(delta_clip) - 1.0) * 100.0
                print(
                    f"Sensitivity gain vs CLIP: {extra_drop:+.4f} absolute, {relative_gain_pct:+.1f}% relative"
                )
            else:
                print(f"Sensitivity gain vs CLIP: {extra_drop:+.4f} absolute, relative=inf")


def build_ref1_graphs_with_prompts():
    variants = [
        ("EXACT", {}),
        ("ALT1", {"subject": "woman"}),
        ("ALT2", {"animal": "dog"}),
        ("ALT3", {"animal": "rabbit"}),
        ("ALT4", {"tshirt_color": "green"}),
        ("ALT5", {"pants_style": "cargo pants"}),
        ("ALT6", {"shoes_style": "brown leather boots"}),
        (
            "ALT7",
            {
                "subject": "woman",
                "animal": "dog",
                "tshirt_color": "black",
                "pants_style": "ripped skinny jeans",
                "shoes_style": "white running shoes",
            },
        ),
        (
            "ALT8",
            {
                "subject": "man",
                "animal": "rabbit",
                "tshirt_color": "green",
                "pants_style": "cargo pants",
                "shoes_style": "brown leather boots",
            },
        ),
        ("ALT9", {"subject": "man", "animal": "dog", "animal_interaction": "walking"}),
    ]

    out = []
    for name, kwargs in variants:
        graph, prompt = build_graph(return_overall_prompt=True, **kwargs)
        out.append((name, graph, prompt))
    return out


def build_ref2_graphs_with_prompts():
    variants = [
        ("EXACT2", {}),
        ("ALT2_1", {"subject": "man"}),
        ("ALT2_2", {"glasses_style": "rectangular glasses"}),
        ("ALT2_3", {"shirt_style": "a fitted white t-shirt"}),
        ("ALT2_4", {"dog_interaction": "walking"}),
        ("ALT2_5", {"scene": "a city sidewalk with buildings and traffic"}),
        ("ALT2_6", {"lighting": "nighttime street lighting"}),
    ]

    out = []
    for name, kwargs in variants:
        graph, prompt = build_graph2(return_overall_prompt=True, **kwargs)
        out.append((name, graph, prompt))
    return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    ref1_image_path = resolve_image_path(data_dir, ["ref1.png"])
    ref2_image_path = resolve_image_path(data_dir, ["ref2.png"])

    ref1_graphs = build_ref1_graphs_with_prompts()
    ref2_graphs = build_ref2_graphs_with_prompts()
    weights = {"node": 1.0, "binding": 1.0, "interaction": 6.0}
    scorer = CLIPScorer().to(device)

    ref1_image = load_image_tensor(ref1_image_path, device)
    with torch.no_grad():
        ref1_image_feat = scorer.encode_image(ref1_image)
    print(f"Using REF1 image: {ref1_image_path}")
    evaluate_graph_set("ref1", ref1_graphs, scorer, ref1_image, ref1_image_feat, weights)

    ref2_image = load_image_tensor(ref2_image_path, device)
    with torch.no_grad():
        ref2_image_feat = scorer.encode_image(ref2_image)
    print(f"\nUsing REF2 image: {ref2_image_path}")
    evaluate_graph_set("ref2", ref2_graphs, scorer, ref2_image, ref2_image_feat, weights)
