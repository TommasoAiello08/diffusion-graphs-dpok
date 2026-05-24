# Diffusion Graphs — Project Documentation

A research project for fine-tuning Stable Diffusion 1.5 with structured (scene-graph) rewards via DPOK (Diffusion Policy Optimization with KL regularization). The codebase is split into:

- **Production DPOK pipeline** (root) — paper-faithful DPOK training with ImageReward, evaluation tooling, and reporting.
- **Earlier research scaffolding** (`src/`) — hand-built reward graphs, hierarchical CLIP-based rewards, FK steering for diffusion, and visualization helpers. Useful background for the reward design but not used in the DPOK production loop.

WE ONLY USE COCO REST IS USELESS
---

## High-level architecture

```
                                  ┌────────────────────────────────────────┐
                                  │  COCO captions                         │
                                  │                                        │
                                  └──────────────────┬─────────────────────┘
                                                     │
                                                     ▼
            ┌────────────────────────────────────────────────────┐
            │ dpok_imagereward.py — paper-faithful DPOK training │
            │                                                    │
            │  Stable Diffusion 1.5 (LoRA on UNet attn projs)    │
            │  + DDIM sampling at training time                  │
            │  + ImageReward terminal reward                     │
            │  + PPO-style IS clipping, K updates per sample     │
            │  + Optional value-function baseline (paper A.5)    │
            │  + W&B logging                                     │
            └──────────────┬──────────────────────┬──────────────┘
                           │                      │
                  LoRA checkpoint        Per-prompt snapshots
                           │                      │
                           ▼                      ▼
            ┌──────────────────────────┐  ┌─────────────────────────────┐
            │ eval_report.py           │  │ dpok_eval_metrics.py        │
            │ Publication-ready report │  │ CLIP / VQA / DSG / T2I-Comp │
            │ from a LoRA path         │  │ on a folder of images       │
            └──────────────────────────┘  └─────────────────────────────┘
```

---

## Root-level files (production DPOK pipeline)

### `dpok_imagereward.py`

The main training script. A faithful implementation of Fan et al. 2023 "DPOK" (arXiv 2305.16381) Online RL setting (Appendix B + A.6).

**What it does.** Loads SD1.5, wraps the UNet in a LoRA adapter, samples trajectories with DDIM, scores final images with ImageReward, and updates the LoRA with PPO-style importance sampling against a frozen reference UNet.

**Key paper choices baked in:**
- Reward = ImageReward (wider dynamic range than CLIP cosine, better SNR for RL).
- CFG is used identically in training-time sampling and in the policy's log-prob (`guidance_scale=7.5` default) — no train/eval mismatch.
- IS ratio clipping with `epsilon=1e-4` (paper eq. 26).
- Each sampled trajectory feeds K=5 gradient steps with n=32 IS pairs per step (paper Appendix B: m=10, n=32).
- AdamW lr=1e-5, alpha=10, beta=0.01, grad-norm clip=0.1.
- Optional value-function baseline (paper A.5) for variance reduction.

**Function-by-function:**

| Function | Role |
|---|---|
| `choose_dtype(name)` | Resolve `auto/fp32/fp16/bf16` → `torch.dtype`. Picks bf16 on Ampere+. |
| `is_finite_tensor(x)` | NaN/Inf guard. Used at every numerical hot-spot during sampling and loss computation. |
| `zero_loss()` | Returns a fresh `requires_grad=True` zero tensor — needed when all samples in a step are skipped, so the optimizer step still has a valid graph. |
| `ValueNetwork` | Small CNN over the 4×64×64 latent + sinusoidal time embedding + pooled prompt embedding → scalar `V(x_t, t, z)`. Trained with MSE against the clipped terminal reward. Optional. |
| `compute_reward(pil_image, prompt)` | Calls the loaded `ImageReward-v1.0` model, returns a scalar fp32 tensor on the current device. |
| `get_sigma_sq(t_val)` | DDIM posterior variance at timestep `t`, with eta and a numerical floor. |
| `noise_pred_to_mu(x_t, noise_pred, t_val)` | Linear map from predicted noise → posterior mean `μ_θ`. Must propagate gradient (the policy log-prob ratio flows through here). |
| `decode_latents_to_pil(latents)` | VAE decode in fp32 with NaN/black-image guards. Returns a PIL image suitable for ImageReward. |
| `sample_trajectory(prompt_embeds, uncond_embeds)` | Run DDIM sampling under the current LoRA UNet with CFG. Returns the per-step `(x_t, x_{t-1}, t, noise_pred_old)` tuples needed for IS, plus the final decoded PIL image. |
| `safe_inference_image(prompt, ...)` | Inference-mode image generation for periodic snapshots. Independent of the training loop. |
| `compute_is_loss_term(...)` | The PPO/DPOK loss for one `(x_t, x_{t-1}, t)` pair. Computes the log-ratio of `p_θ` to `p_θ_old` from the difference of squared deviations to `μ_new` vs `μ_old`, clips the ratio to `[1-ε, 1+ε]`, multiplies by the (optionally advantage-baselined) reward, adds the KL-to-reference term, and adds the value-function MSE term. Returns `(loss, stats)`. |

**Training loop logic (lines 661–873):**
1. Sample `m=10` trajectories from the current LoRA UNet → pool every `(x_t, x_{t-1}, t)` step from every trajectory.
2. Run `K=5` gradient updates. Each step: sample `n=32` pairs with replacement, compute the average IS loss, clip grads to 0.1, step AdamW.
3. Every `--save_every` rounds: save LoRA checkpoint + state.pt, render a per-prompt snapshot with fixed seeds, log to W&B.

**Outputs:**
- `lora_unet_final/` — final LoRA weights.
- `checkpoints/round_NNNNN/` — periodic LoRA + optimizer state.
- `inference/round_NNNNN/prompt_NN.png` — per-prompt snapshot grid.
- `snapshots.csv`, `per_prompt_curves.png`, `snapshot_grid.png` — training trajectory visualizations.
- `reward_curve.png` — round-level mean reward over training.

---

### `dpok_scene_reward.py`

Self-contained data loaders + reward models for DPOK training. **No `src/` imports** — designed to be importable in a clean HPC env that only needs PyTorch + open_clip + transformers.

**Data classes:**

| Class | Purpose |
|---|---|
| `SceneGraph` | `(objects, attributes, relations)` triple. Lightweight container for graph-level evaluation. |
| `PromptItem` | `(text, object_text, relation_text, graph, source, image_id)`. One training/eval row. The `text` field is the prompt fed to SD; the `graph` field is used by hierarchical reward variants and decomposed eval. |

**Data loaders:**

| Function | What it reads → returns |
|---|---|
| `load_coco_items(json_path, max_captions, seed)` | COCO captions JSON → list of `PromptItem` with no graph. |
| `load_psg_items(json_path, max_scenes, max_rel, seed)` | OpenPSG annotations → `PromptItem` with object list + relation triples. Builds prompts like `"A scene where X pred Y, ..."`. |
| `load_gqa_items(json_path, max_scenes, max_objects, max_rel, seed)` | GQA scene graphs → `PromptItem` with objects, attributes, and relations. Sorts objects by bbox area. |
| `load_coco_psg_items(coco_path, psg_path, ...)` | Joins COCO captions with PSG graphs on `image_id`. Returns items whose `text` is a real COCO caption and whose `graph` is the matching PSG scene graph — the dataset of choice when you want both human language and structured graph supervision. |

**Internal helpers:** `_clean_name`, `_psg_obj_name`, `_gqa_short_desc`, `_gqa_sorted_objects`, `_gqa_collect_relations` — small utility functions for normalizing names and reshaping the raw JSON.

**Reward classes:**

| Class | What it scores |
|---|---|
| `RewardModel` | Plain global CLIPScore (ViT-L-14, OpenAI). `reward = (cos_sim − baseline) × scale`. Default `baseline=0.20`, `scale=10.0`. Has `compute()` for training and `compute_detailed()` for eval (returns per-object / per-relation / per-attribute scores from the graph). |
| `BLIPRewardModel` | BLIP image-text matching reward (`Salesforce/blip-itm-base-coco`). Outputs `P(image matches text) ∈ [0,1]`; mapped to `(p − 0.5) × 10` for a roughly ±5 reward range. About 6× wider dynamic range than CLIP-ViT-B-32 cosine. Used as the BLIP-ITM baseline in the reward comparison and also by downstream code. |
| `HierarchicalRewardModel` | Weighted combination of global CLIP + mean per-object CLIP + mean per-relation CLIP + mean per-attribute CLIP, then `(combined − baseline) × scale`. Auto-renormalizes weights when the graph is missing relations or attributes. Default weights `0.30 / 0.35 / 0.25 / 0.10`. |

Each reward model has:
- `__init__(device, baseline, scale, ...)` — loads the backbone once, freezes parameters.
- `compute(image, item)` / `compute(image, prompt)` — returns a scalar fp32 tensor on `device`.
- `sanity_check()` — smoke test with a blank image.

---

### `dpok_eval_metrics.py`

Standalone evaluation script for a folder of pre-generated images. Does **not** generate images — designed to be run after `dpok_imagereward.py` or any inference script has already produced outputs.

**Metrics implemented (each is optional):**

| Scorer class | Backbone | Returns |
|---|---|---|
| `CLIPScorer` | open_clip `ViT-L-14/openai` | Cosine similarity between image and full prompt. |
| `ImageRewardScorer` | `image-reward` package, model `ImageReward-v1.0` | Scalar reward in roughly `[-2, +2]`. Wraps PIL → temp PNG for the library. |
| `VQAScorer` | `t2v-metrics` package, `clip-flant5-xl` | `P("Does this image show <prompt>?")` from CLIP-FlanT5. |
| `DSGScorer(vqa_scorer)` | uses `VQAScorer` internally | Mean VQA score over atomic yes/no claims extracted from the prompt. |
| `T2ICompScorer(vqa_scorer)` | uses `VQAScorer` internally | T2I-CompBench style per-category (object/attribute/relation) scores. |

**Helper functions:**

| Function | Role |
|---|---|
| `extract_atomic_claims(prompt, max_claims)` | Rule-based decomposition into yes/no questions: `"Is there a <noun>?"`, `"Is the <noun> <adj>?"`, `"Is <X> <prep> <Y>?"`. Used by `DSGScorer`. Lightweight replacement for the LLM-based DSG decomposition. |
| `generate_t2i_questions(prompt)` | Same idea but split into three buckets (`object`, `attribute`, `relation`) for T2I-CompBench style scoring. |
| `plot_score_distributions(per_image, save_path, baseline_data)` | Side-by-side histograms per metric, with mean lines. Optionally overlays a baseline run. |
| `print_summary(aggregate, baseline_agg)` | Formatted terminal table with `Trained / Baseline / Delta` columns. |
| `parse_args()` / `main()` | CLI wiring. Reads `--image_dir`, `--prompts_file`, optional `--baseline_json` for A/B comparison, and method toggles `--skip_clip / --skip_vqa / --skip_dsg / --skip_t2i`. |

**Outputs:** `metrics.json` (per-image + aggregate), `summary.txt`, `score_distributions.png`.

---

### `eval_report.py`

A higher-level wrapper that **generates** images from a (LoRA, prompt-list) pair and then runs the same metric stack as `dpok_eval_metrics.py`. Designed for producing publication-ready evaluation outputs:

- `report.txt` — formatted ASCII table.
- `report.json` — machine-readable scores.
- `report.tex` — LaTeX table.
- `per_image_scores.csv` — per-image breakdown.
- `score_distributions.png`, `score_radar.png` — plots.

**Key entry points:**

| Function | Role |
|---|---|
| `parse_args()` | CLI: `--lora_path`, `--coco/--psg/--gqa` or `--prompts_file`, `--metrics clip,vqa,dsg,t2i`, `--seeds_per_prompt`. |
| `generate_images(pipe, prompts, num_steps, guidance_scale, seeds, save_dir)` | Runs SD1.5 inference per prompt × seed. Saves PNGs to disk. |
| Per-metric callers | Wrap the `CLIPScorer / VQAScorer / DSGScorer / T2ICompScorer` from `dpok_eval_metrics.py`. |
| Reporting | `format_table()`, `write_latex()`, `make_radar()` — turn the aggregate dict into the four output files. |

The `--prompts_file` path is the "paper-style" mode: render each prompt with `--seeds_per_prompt` (default 10) independent seeds, then score the whole set. This matches how the original DPOK paper reports numbers (4 curated compositional prompts × many seeds).

---

## `src/` — research baseline and supporting code

Older but still-useful exploration code that the DPOK pipeline draws design ideas from. Not imported by the production scripts.

### `src/rewards/clip.py`

A bare CLIPScorer module (`open_clip` ViT-L-14 by default).

| Function | Role |
|---|---|
| `CLIPScorer.encode_text(text)` | Tokenize + encode a single text → unit-normalized text feature. |
| `_interpolate_positional_embedding(pos_embed, grid_h, grid_w)` | Bicubic-interpolate CLIP's positional embeddings when running on non-standard image grids. Lets the scorer accept variable-resolution images. |
| `_encode_image_variable_resolution(image_tensor)` | Forward pass through CLIP visual transformer with the interpolated positional embedding. Returns the projection-pooled feature. |
| `encode_image(image_tensor)` | Public wrapper around the above; returns unit-normalized image feature. |
| `score_texts_from_image_feature(image_feat, texts)` | Batched cosine similarity between one image feature and many texts. |
| `score_from_image_feature(image_feat, text)` | Single-text version of the above. |
| `score(image_tensor, text)` | One-call: encode image, encode text, return cosine. |

The variable-resolution image encoding is what differentiates this scorer from the simpler `CLIPScorer` in `dpok_eval_metrics.py` — it supports gradient flow through arbitrary image sizes, which matters when you want the reward to be differentiable with respect to a learnable latent.

### `src/rewards/rewards.py`

The original hand-built hierarchical reward — the inspiration for `HierarchicalRewardModel` in `dpok_scene_reward.py`.

| Dataclass | Fields |
|---|---|
| `Node` | `name`, `exist_prompt`, `neg_prompts` — one graph node + the texts used for its existence probability and negative contrasts. |
| `Binding` | `node` (name), `prompt`, `neg_prompts` — an attribute or appearance phrase tied to a node (e.g. node `man`, binding `"a white t-shirt"`). |
| `Interaction` | `src`, `dst`, `prompt`, `neg_prompts` — a relation between two nodes. |
| `Graph` | `nodes`, `bindings`, `interactions`. |

**`SimpleHierReward(torch.nn.Module)`** — gated hierarchical reward:

| Method | Role |
|---|---|
| `__init__(scorer, graph, weights, gating, alpha, gate_tau, contrastive_temp, contrastive_mode, hard_threshold)` | Configure weights for nodes / bindings / interactions, gating mode (`soft` sigmoid or `hard` threshold), and contrastive scoring mode (`prob` softmax or `margin`). |
| `_gate(phi_node)` | Either `sigmoid(α·(φ − τ))` (soft) or `(φ > thr).float()` (hard, detached). The bindings and interactions for a node are multiplied by its gate so absent nodes don't contribute. |
| `_contrastive_potential(image_feat, pos_prompt, neg_prompts)` | Contrast the positive prompt against negatives — either as softmax probability (`prob` mode) or as `pos − temp·logsumexp(neg/temp)` (`margin` mode). This is the key idea that the new Grounded SG-reward's antonym contrast inherits. |
| `forward(image)` | Encode image once. Compute node / binding / interaction potentials in batches. Apply gating. Return `(R, dbg)` — the scalar reward and a debug dict of intermediate tensors for the visualizer. |

There is also a `RewardGraph` class in the same file (referenced from `message_passing_clip_reward.py` and `example.py`) — a message-passing variant where gate values propagate through binding/interaction edges. Same dataclasses, different aggregation.

### `src/refgraphs.py` and `src/refgraphs2.py`

Hand-written reference scene graphs used by the older example scripts. They define a small zoo of `(Node, Binding, Interaction)` triples plus prompt-generation helpers:

- Constants: `PRONOUN`, `SUBJECTS`, `ANIMALS`, `TSHIRT_COLORS`, `PANTS_STYLES`, `SHOES_STYLES`, etc.
- `_exclude(options, selected)`, `_unique(values)` — list utilities for building negative prompts.
- `_subject_glasses_prompt(subject, glasses_style)` — phrase generator.
- `build_graph(...)` and `build_graph2(...)` — return `(graph_name, Graph, prompt_text)` triples used by the example scripts to systematically vary the scene description.

These exist purely as test fixtures — they let the older scripts exercise the hierarchical reward on a closed, controllable set of prompts (one subject, one outfit, optional animal, optional glasses).

### `src/simple_hier_clip_reward.py` and `src/message_passing_clip_reward.py`

Driver scripts that load an image, build several reference graphs, evaluate them with `SimpleHierReward` (or `RewardGraph`), and print a comparison.

| Function | Role |
|---|---|
| `load_image_tensor(image_path, device)` | Load PIL → unit-tensor `(1, 3, H, W)` with `requires_grad=True` (so the reward is differentiable with respect to the image). |
| `resolve_image_path(data_dir, preferred_names)` | Look up the first matching filename — supports a small known set of demo images. |
| `compute_plain_clip_score(scorer, image_feat, full_prompt)` | Plain cosine similarity, for the "baseline" column. |
| `format_graph_summary(graph_name, graph, full_prompt)` | Pretty-print a graph (node count, binding count, interaction list). |
| `evaluate_graph_set(set_name, graphs, scorer, image, image_feat, weights)` | For each graph in the set, build a reward model, run it, and print baseline-relative deltas. The baseline graph is the one named `EXACT` / `EXACT2`; all others are perturbations. |

The two files differ only in which reward class they instantiate (`SimpleHierReward` vs `RewardGraph`). Conceptually this is the same idea as the "score perturbed scene graphs against the same image" experiment but on the older hand-built graphs.

### `src/example.py`, `src/example_t2i.py`, `src/example_fksteering.py`

Minimal worked examples:

- `example_t2i.py` — bare SD1.5 inference. No reward, no steering. Used as a sanity check that the pipeline loads.
- `example.py` — SD1.5 + `RewardGraph` over the `EXACT` reference graph from `refgraphs.py`. Demonstrates how to feed the reward signal back into the denoising process.
- `example_fksteering.py` — FK steering (see below) wrapped around the same SD1.5 + reward stack.

The shared function `sample_with_reward(num_steps, alpha, save_steps, steps_dir)` in `example.py` is the prototype that DPOK's `sample_trajectory()` evolved from — it shows the simplest version of "sample with diffusion, score with reward, push gradient back."

### `src/viz/viz_graph.py`

Visualization of a reward graph as a Graphviz PNG.

| Function | Role |
|---|---|
| `render_reward_graph_weighted(nodes, bindings, interactions, dbg, weights, out_path, keep_dot)` | Build a Graphviz `Digraph` with one ellipse per node, one box per binding, one arrow per interaction. Width/color of each element scales with its contribution from the `dbg` dict (which `SimpleHierReward.forward` returns). Renders to PNG via the `dot` binary. |
| `_to_list(x)` / `_vec(size, *keys, default)` | Tensor-to-list and key-aliasing helpers so the renderer is tolerant of which exact keys the reward model emitted. |

Requires the system `graphviz` binary and the `graphviz` Python package.

### `src/fk_steering/fkd_diffusers/`

A self-contained implementation of Feynman-Kac diffusion (FKD) particle-based steering. Lets a frozen SD1.5 pipeline be guided toward high-reward regions of latent space at inference time without finetuning — a complement to DPOK (no gradient updates to UNet, just particle resampling).

| File | Contents |
|---|---|
| `fkd_class.py` | `FKD` class with `PotentialType` enum (`DIFF`, `MAX`, `ADD`, `RT`). Maintains `num_particles` latent populations; at each timestep computes per-particle rewards, then resamples particles weighted by `exp(λ · potential)`. Supports adaptive resampling and `t_start/t_end` windowing. |
| `fkd_pipeline_sd.py` | A `StableDiffusionPipeline` subclass that calls `FKD.resample()` after each denoising step. Drop-in replacement for the vanilla pipeline. |
| `rewards.py` | Reward registry `REWARDS_DICT` with handles for `Clip-Score`, `ImageReward`, `LLMGrader`, and the project's own `Project-CLIPScorer` / `Project-RewardModel`. Lazy-imports each backend so missing packages don't crash unrelated paths. |
| `image_reward_utils.py` | Thin wrappers around `ImageReward.load()` + scoring helpers used by the registry. |

FK steering is an alternative to gradient-based RL: instead of updating model weights, it keeps a particle population in the latent space and biases sampling toward high-reward particles. Cheap, but no learning persists across calls. The DPOK pipeline took the reward-design ideas (hierarchical + graph-aware) from this branch and put them into a gradient-update loop.

---

## Two ways to think about the codebase

**By dependency direction:** root-level production scripts (`dpok_*.py`, `eval_report.py`) are self-contained and don't import from `src/`. The `src/` tree is older research code with its own internal imports (`from rewards.clip import CLIPScorer`, etc.). You can throw away `src/` and the DPOK training loop still works.

**By concept:** every reward in this codebase scores `(image, text)` and returns a scalar:

- **Flat scorers** (`CLIPScorer`, `BLIPRewardModel`, `ImageRewardScorer`, `VQAScorer`) — one number per pair, no structure.
- **Decomposed scorers** (`HierarchicalRewardModel`, `SimpleHierReward`, `RewardGraph`, `DSGScorer`, `T2ICompScorer`) — break the prompt into pieces, score each, aggregate. They differ in *how* they decompose: hand-built graph (`SimpleHierReward`), structured `SceneGraph` field (`HierarchicalRewardModel`), regex over the prompt (`DSGScorer`), or LLM-style decomposition (full DSG paper). The trade-off is always "more structure = more compositional sensitivity = slower."

The DPOK training loop in `dpok_imagereward.py` deliberately uses the simplest one (ImageReward, flat) because the RL signal needs to be cheap. The evaluation scripts use the slowest ones (VQA, DSG) because eval is offline and can spend time. This split is the core engineering trade-off the project navigates.
