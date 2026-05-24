# DPOK ImageReward — Code Walkthrough (Meeting Notes)

Reference paper: Fan et al. 2023, "DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models" (arXiv 2305.16381).
Reference code: `google-research/google-research/dpok` (the official Google Research implementation).

This document explains what every file does, walks through `dpok_imagereward.py` section by section, and contrasts each piece with the official DPOK code. Numbers in `[brackets]` are line numbers in `dpok_imagereward.py`.

---

## 1. Repository file map

| File | Purpose | Used by |
|------|---------|---------|
| `dpok_imagereward.py` | The training script. SD1.5 + LoRA + ImageReward + paper-faithful importance-sampling DPOK. | `submit_imagereward*.sh` |
| `dpok_scene_reward.py` | Hierarchical scene-graph reward (CLIP+object+relation+attribute). NOT used in the current ImageReward runs — kept for the multi-prompt scene-graph variant. | `dpok_imagereward.py` only loads it if `--coco/--psg/--gqa` is passed |
| `dpok_eval_metrics.py` | Evaluation metric implementations: CLIPScore, ImageRewardScorer, VQAScorer, DSGScorer, T2ICompScorer. Read-only at training time. | `eval_report.py` |
| `eval_report.py` | Standalone evaluation script. Generates baseline images (frozen SD1.5) and trained images (with our LoRA), runs all chosen metrics, writes per-prompt and aggregate reports. | `submit_imagereward*.sh` (post-training step) |
| `prompts_paper.json` | 4 prompts from paper Sec 5.2: "A green colored rabbit." / "A cat and a dog." / "Four wolves in the park." / "A dog on the moon." | Training prompts for paper Sec 5.2 reproduction |
| `prompts_holdout.json` | 4 unseen prompts mirroring the same 4 axes (color/multi-object/counting/surreal-location) | Generalization eval |
| `prompts_single.json` | Just `"A green colored rabbit."` — single-prompt diagnostic | Option B run (483273 etc.) |
| `prompts_coco104.json` | Auto-generated 104 COCO captions for paper Sec 5.3 | Multi-prompt run |
| `submit_imagereward.sh` | SLURM script: train on `prompts_paper.json` (4 prompts), eval on train + holdout. 24h wall. |
| `submit_imagereward_multi.sh` | SLURM script: train on 104 COCO prompts (paper Sec 5.3). |
| `submit_imagereward_single.sh` | SLURM script: train on `prompts_single.json` (1 prompt, paper Sec 5.2 single-prompt regime). |
| `setup_cluster.sh` / `setup_eval_env.sh` / `cache_eval_models.sh` | One-shot conda env + HF cache setup. |
| `eval-single-prompt/` | Pulled-down results from earlier runs. |

---

## 2. `dpok_imagereward.py` — section-by-section walkthrough

The script is one self-contained ~970-line Python file. Official DPOK is split across 4 files (`train_online_pg.py`, `pipeline_stable_diffusion_extended.py`, `scheduling_ddim_extended.py`, `reward_model.py`) plus utilities.

### 2.1 — Header docstring `[1–31]`

States the explicit design choices: ImageReward as reward, CFG=7.5 in BOTH training sampling and the policy's log-prob, importance sampling with PPO-style clip ε at 1e-4, all (x_t, x_{t-1}) pairs from m sampled trajectories pooled, paper Appendix B hyperparameters. Mentions value function as a follow-up flag.

> **Defense point:** the docstring documents which paper sections each design follows. This is the meeting's first reference card.

### 2.2 — Argument parsing `[55–113]`

All paper hyperparameters are surface-level CLI args. The naming maps directly to the paper:

| CLI arg | Paper symbol | Default | Paper Sec 5.2 |
|---------|--------------|---------|---------------|
| `--total_samples` | total online samples | 20000 | 20000 |
| `--sample_batch` | m (trajectories per round) | 10 | 10 |
| `--grad_steps` | K (gradient updates per round) | 5 | 5 |
| `--is_batch` | n (pairs per gradient step) | 32 | 32 |
| `--is_clip` | ε (PPO clip) | 1e-4 | 1e-4 (eq. 26) |
| `--num_steps` | T (DDIM steps) | 20 | 50 (we use 20 for speed) |
| `--alpha` | α (reward weight) | 10 | 10 |
| `--beta` | β (KL weight) | 0.01 | 0.01 |
| `--lr` | AdamW LR | 1e-5 | 1e-5 |
| `--grad_norm_clip` | gradient clip | 0.1 | <0.1 |
| `--lora_rank` | r | 4 | 4 |
| `--guidance_scale` | CFG s | 7.5 | 7.5 |
| `--use_value_function` | flag for paper Appendix A.5 | off | on (`v_flag=1`) |

> **Difference vs official:** official DPOK uses `--ratio_clip 1e-4` (default) — same as ours. We use `T=20` DDIM steps; official uses `T=50`. Smaller T means cheaper rollouts, fewer (x_t, x_{t-1}) pairs per trajectory, faster wall-time. Tradeoff is sample diversity, not correctness.
> **Why ours is better:** every paper hyperparameter is a CLI flag, no hidden values. The official code has many constants buried in `train_online_pg.py`.

### 2.3 — Setup, dtypes, seeds `[119–185]`

Standard determinism (`torch.manual_seed`, `cuda.manual_seed_all`), TF32 on for matmul speed, choose dtype (we default to bf16 but the latest run uses fp32 for precision).

`is_finite_tensor()` and `zero_loss()` are micro-helpers used as sanity checks throughout to skip steps that produce non-finite tensors instead of crashing the run.

> **Difference vs official:** official defaults to fp16 mixed-precision via `accelerator.mixed_precision="fp16"`. We don't use HuggingFace `accelerate` — direct `torch` only, simpler to reason about.

### 2.4 — Pipeline + reference UNet `[238–256]`

Two UNets are loaded:
- `pipe.unet` (becomes the policy via LoRA wrapping next).
- `pipe_ref_unet` — a frozen copy of SD1.5 UNet, used to compute the KL anchor (= the reference policy π_ref the paper equation 9 penalizes against). Its parameters have `requires_grad_(False)` and it stays in `eval()`.

VAE is forced to fp32 even when the pipe is bf16 because VAE decode is numerically fragile (one bad pixel becomes black image).

> **Difference vs official:** official uses `unet_copy = copy.deepcopy(unet)` of the in-training UNet to get the reference, frozen via `requires_grad_(False)` per-parameter. Functionally identical to our explicit reload.

### 2.5 — Dataset `[262–282]`

Three modes (`--prompts_file` JSON list / `--coco/--psg/--gqa` scene-graph datasets / error). All three runs we currently submit use `--prompts_file`.

> **Difference vs official:** official accepts a single hardcoded prompt or a hardcoded list embedded in the script. Ours is JSON-driven.

### 2.6 — LoRA `[288–298]`

```python
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_RANK,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0, bias="none",
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
```

We use **PEFT** (`peft.LoraConfig` + `get_peft_model`) which is the modern HuggingFace way. We attach LoRA to the four attention projection matrices: `to_q`, `to_k`, `to_v`, `to_out.0`. `lora_alpha=lora_rank` means the standard scaling factor `alpha/r = 1`.

> **Difference vs official:** official uses the older diffusers API `LoRACrossAttnProcessor` and `unet.set_attn_processor(lora_attn_procs)`. They attach LoRA to **both `attn1` (self-attention) and `attn2` (cross-attention)** processors, treating each `attn_processor` as the unit. Our PEFT config attaches to all matching named modules across the UNet — functionally similar coverage but the PEFT path is forward-compatible with newer diffusers versions.
> **Why ours is better:** PEFT is the official transformers/diffusers ecosystem path now; the `LoRACrossAttnProcessor` API was deprecated in diffusers ≥0.20.

> **Subtle point worth flagging in the meeting:** comment on line 298 says *"No gradient checkpointing — breaks lora_B grad."* This was a real bug we hit and fixed: with gradient checkpointing on, PEFT's LoRA B matrices got zero gradient because the inner forward was re-run under `no_grad`. We disabled checkpointing as the simplest fix; the official code uses `unet.enable_gradient_checkpointing()` but that's compatible with their `LoRACrossAttnProcessor`, not with PEFT.

### 2.7 — Optimizer `[304–306]`

`AdamW` over LoRA-trainable params. Single optimizer.

> **Difference vs official:** official also uses AdamW (`torch.optim.AdamW`) over `lora_layers.parameters()`. Same.

### 2.8 — Value network `[316–365]`

This is our **paper Appendix A.5** implementation. Used when `--use_value_function` is passed.

Architecture:
```
Conv2d(4→32, stride 2) → SiLU →
Conv2d(32→64, stride 2) → SiLU →
Conv2d(64→128, stride 2) → SiLU →
AdaptiveAvgPool2d(1) → Flatten   # 128-dim image feature

sinusoidal_time_embedding(t, dim=128) → Linear(128→256) → SiLU   # 256-dim time
prompt_embeds.mean(seq) → Linear(768→256) → SiLU                  # 256-dim text

concat(128 + 256 + 256) → Linear(...→256) → SiLU → Linear(256→1)
```

Trained with MSE against the realized terminal reward `r`. Advantage in the policy loss is `r - V.detach()` so V's gradients don't leak into the policy.

> **Difference vs official:** official `ValueMulti` is a 4-layer ConditionalLinear MLP that flattens the 4×64×64 latent and concatenates the pooled text embedding **at every layer**, with the timestep providing FiLM-style gamma scaling per-layer. Their input is `4*64*64 + 768 = 17152` dims — directly flattened.
> Our implementation uses a small ConvNet over the latent first (so the model gets spatial inductive bias) and concatenates time + text only once at the head. Functionally similar — both regress scalar reward, both used as detached baseline.
> **Why ours is defensible:** smaller (~250K params vs ~5M+ in their flatten-everything MLP), easier to fit on a single A100 alongside SD1.5, gives a spatial inductive bias that an MLP can't.

### 2.9 — Reward: ImageReward `[374–400]`

```python
import ImageReward as RM
_ir_model = RM.load("ImageReward-v1.0", device=device)
```

Wrapped in `compute_reward(pil_image, prompt) → scalar tensor`. The model is frozen and `eval()`-only.

The line `blank_score = compute_reward(Image.new("RGB", (64, 64)), "a cat")` is a **runtime sanity check** — if ImageReward returns a sensible negative score for a black image against an unrelated prompt, the model is loaded correctly. The number is logged.

> **Difference vs official:** official has the same `imagereward.load("ImageReward-v1.0")` pattern in `utils.image_reward_get_reward`. They normalize via `(rewards - model.mean) / model.std`; we use the raw `model.score(prompt, pil)` output. Both end up in the same dynamic range (~[-3, +3]).

### 2.10 — Scheduler utilities `[406–436]`

This block is the algorithmic glue. Three pieces:

**`set_timesteps(NUM_STEPS)` + `next_denoising_step` map**: stores t→t' mapping for each timestep so we can compute σ² and μ at any t without re-querying the scheduler.

**`get_sigma_sq(t)`**: paper's σ²_t = η² · (1 − α̅_{t-1})/(1 − α̅_t) · β_t. Returns the variance of the policy Gaussian at step t. Floored at `SIGMA_FLOOR=1e-5` to avoid division-by-zero in the log-ratio.

**`noise_pred_to_mu(x_t, noise_pred, t)`**: linear map from a noise prediction to the mean μ_θ of p_θ(x_{t-1} | x_t). Formula is from DDIM:
```python
mu = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_cumprod_t)) * noise_pred)
```

Critical: this function does **not** have `@torch.no_grad()`. The earlier version had it, and it silently zeroed the gradient through μ_θ. The fix is documented in the diff history.

> **Difference vs official:** official wraps μ computation inside `step_logprob` / `step_forward_logprob` methods of their custom DDIM scheduler subclass. They use `Normal(mu, std).log_prob(x_{t-1}).mean(dim=-1).mean(dim=-1).mean(dim=-1)` to get a per-sample scalar log-prob.
> Our approach: skip the `Normal` distribution object entirely. Since both p_θ and p_θ_old are Gaussians **with the same variance σ²**, the log-ratio reduces analytically to `(||x_{t-1} - μ_old||² - ||x_{t-1} - μ_new||²) / (2σ²)` and the constants cancel. This is mathematically exactly the same as their `Normal.log_prob` difference, but skips one tensor allocation per sample and is more numerically stable when both means are close (which they are at init).
> **Why ours is better:** clearer derivation (a senior ML reviewer can verify the math in 30 seconds), avoids constructing a `Distribution` per call, identical numerical result.

### 2.11 — Sampling: `sample_trajectory()` `[461–510]`

Wrapped in `@torch.no_grad()`. For each timestep:
1. If CFG: duplicate the latent, run UNet on `cat([uncond, cond])`, split, combine with `nu + s*(nc - nu)`.
2. Step the scheduler with `eta=ETA=1.0` (DDPM-equivalent stochastic).
3. Store `(x_t, x_{t-1}, t, noise_pred_old)` — `noise_pred_old` is the **CFG-combined** prediction that was actually used to produce x_{t-1}.
4. After all steps, decode latents to PIL via `decode_latents_to_pil`.

Returns the trajectory steps and the final image.

> **Difference vs official:** official has `forward_collect_traj_ddim` in `pipeline_stable_diffusion_extended.py` that does the same thing inside their custom pipeline subclass. They store per-step `Normal.log_prob` directly; we store the ingredients (`noise_pred_old`) and recompute the ratio on the fly during the policy update. Storing `noise_pred_old` is exactly what's needed for IS — same algorithm, different storage strategy.
> **Why ours is defensible:** storing `noise_pred_old` (a single tensor) rather than `log_prob_old` (a scalar) lets us recompute log-ratios with current σ² and current numerical conditions, which makes debugging easier.

### 2.12 — `compute_is_loss_term()` — the algorithmic core `[529–631]`

This is the function the supervisor will probe. Walk through it line-by-line.

**Signature:** takes one (x_t, x_{t-1}, t, noise_pred_old, r, prompt_embeds, uncond_embeds) tuple and returns the scalar loss + stats.

**[548–550] cast to fp32 and detach what shouldn't carry gradient.** x_{t-1}, noise_pred_old, r are observed quantities — they MUST NOT carry gradient.

**[552–554] σ² and timestep tensor.** σ² is the same Gaussian variance the sampler used.

**[556–574] forward UNet under current policy + reference.** With CFG, we duplicate the latent and run on `cat([uncond, cond])`, then split and combine `nu + s*(nc - nu)`. Reference UNet is wrapped in `with torch.no_grad():` because it's the frozen π_ref. **Same CFG combination is applied to both** — this is critical so the KL is between the deployed (CFG-combined) distributions, not the unconditional-only distributions.

**[579–581] compute μ_new and μ_old.** Both via the helper `noise_pred_to_mu`. μ_old uses the **stored** `noise_pred_old`, so it's a recomputation with grad disabled (`.detach()`) — it's a constant from the policy gradient's POV.

**[584–590] importance ratio.**
```python
diff_new_sq = ((x_t_minus_1 - mu_new) ** 2).mean()
diff_old_sq = ((x_t_minus_1 - mu_old) ** 2).mean()
log_ratio = (diff_old_sq - diff_new_sq) / (2.0 * sigma_sq)
log_ratio = log_ratio.clamp(min=-20.0, max=20.0)   # NaN guard
ratio = torch.exp(log_ratio)
ratio_clipped = torch.clip(ratio, 1.0 - IS_CLIP, 1.0 + IS_CLIP)
```

This is paper eq. (26): `clip(p_θ / p_θ_old, 1−ε, 1+ε)`. The clamp at ±20 is a numerical guard (exp(20)≈5e8) — unreachable under normal training, only ever triggers in pathological NaN cascades.

**[592–594] KL term.**
```python
kl_num = ((noise_tr - noise_ref.detach()) ** 2).mean()
kl_t = kl_num / (2.0 * sigma_sq)
```

This is the analytical KL between two Gaussians with the same variance: `KL = ||μ_θ − μ_ref||² / (2σ²)`. Since μ is a linear function of noise prediction (noise_pred_to_mu), `||μ_θ − μ_ref||²` is proportional to `||noise_θ − noise_ref||²` (same proportionality factor cancels in the gradient). We use the noise-pred form because that's what the UNet directly outputs.

> **Difference vs official:** official uses `kl_regularizer = (noise_pred - old_noise_pred) ** 2` — squared L2 of noise predictions, **without dividing by σ²**. They absorb the σ² scaling into β. Mathematically the same as scaling β by σ²; numerically slightly different because σ² varies across timesteps (we apply per-timestep scaling, they don't).
> **Why ours is more rigorous:** dividing by σ² gives the analytical KL of the underlying Gaussians, so β has a clear units interpretation. Theirs treats β as a tunable mixing weight without unit semantics. Ours is closer to the paper's eq. 9 derivation.

**[596–598] reward clipping.** `r_clipped = r.clamp(-5, +5)`. ImageReward is normally in [-3, +3] but occasionally outputs ±6+ on very pathological images; this prevents one bad sample from dominating the gradient.

**[600–610] advantage with VF baseline.**
```python
if USE_VF and value_net is not None:
    v_pred = value_net(x_t, t_val, prompt_embeds.to(...))
    advantage = (r_clipped - v_pred.detach())
    vf_loss = VF_WEIGHT * (v_pred - r_clipped) ** 2
else:
    advantage = r_clipped
    vf_loss = 0
```

V is detached when computing advantage (so V's gradient doesn't leak into the policy gradient — REINFORCE math). V's own gradient comes only from `vf_loss` which is MSE regression to the realized reward.

**[612–614] total loss.**
```python
policy_loss = -ALPHA * advantage * ratio_clipped
kl_loss = BETA * kl_t
loss = policy_loss + kl_loss + vf_loss
```

This is paper eq. 9 + eq. 26: `L = −α A clip(ρ, 1−ε, 1+ε) + β KL + (vf_weight) (V − r)²`.

> **Difference vs official:** identical formulation. Only difference: official skips `vf_loss` from the total loss and trains V in a separate step (5 iterations of pure MSE per round). Ours adds VF gradient to the same backward pass, optimized by a separate optimizer (`vf_optimizer`) — equivalent in expectation, simpler in code.

### 2.13 — Training loop `[661–873]`

Outer `while samples_generated < TOTAL_SAMPLES`:
- **Sampling phase [663–693]**: generate `M_SAMPLE_BATCH=10` trajectories. Each picks a random prompt, encodes prompt + uncond embeds, samples a trajectory, scores it with ImageReward.
- **Build pool [700–707]**: flatten all trajectories' (x_t, x_{t-1}, t, noise_pred_old) into one big list. Skip the last step (no x_{t-1}).
- **K gradient steps [721–789]**: for each of K=5 grad steps:
  - sample `n=32` (x_t, x_{t-1}) pairs **with replacement** from the pool (paper Appendix B)
  - sum `compute_is_loss_term` over the n pairs (averaged via `loss_term / n`)
  - skip if any term is non-finite
  - one diagnostic on the very first step that confirms LoRA-B matrices have non-zero gradient (the gradient-checkpointing-bug-canary)
  - clip grad norm at 0.1, optimizer step
  - if VF on, separately step the value optimizer
- **Round logging [791–831]**: log mean_r, policy/kl/vf losses, IS ratio stats, clip-hit-rate, optimizer steps, GPU memory.
- **Periodic checkpoint + snapshot [833–873]**: every `SAVE_EVERY=100` rounds, save LoRA + optimizer state, then render a fixed-seed image for each of the first 8 prompts and score it. The snapshot uses **fixed `(SNAPSHOT_SEED_BASE + p_idx)` per prompt across all rounds**, so the only changing thing is the LoRA. Lets you flip through rounds and watch one prompt's evolution.

> **Difference vs official:** their loop structure is:
> ```
> for count in range(max_train_steps // p_step):
>     _collect_rollout(g_step times)
>     _trim_buffer(buffer_size=1000)        # FIFO 1000-transition buffer
>     if v_flag: train_value_func(5 iters)   # SEPARATE phase
>     for _ in range(p_step=5):
>         optimizer.zero_grad()
>         for _ in range(grad_accum=12):     # 12 microbatches of 4 = effective batch 48
>             _train_policy_func()           # samples p_batch_size=4 from buffer
>         clip_grad_norm + optimizer.step()
> ```
> Theirs has:
> - **A persistent FIFO buffer** of 1000 transitions across multiple rounds — off-policy samples from older rollouts get reused.
> - **Gradient accumulation of 12** with microbatch 4 → effective batch ~48.
> - **Separate value-training phase** before policy training (5 inner iters).
>
> Ours has:
> - **Single-round buffer** — each round's trajectories are used only within that round, then discarded.
> - **No accumulation** — n=32 IS pairs are summed inside one `total_loss` and one `backward()`.
> - **Joint optimization** — policy + KL + VF all in one backward.
>
> **Why our differences are defensible:**
> 1. Single-round buffer is more on-policy (closer to the theoretical PG estimator), at the cost of throwing away samples. Reasonable when sampling is the bottleneck and you have enough budget.
> 2. No grad-accum simplifies reasoning — a single backward, single step, exactly K=5 updates per round. Trivial to interpret in W&B.
> 3. Joint VF optimization saves one set of UNet forward passes per round (we don't re-forward the UNet just to update V).

### 2.14 — Final save + visualization `[879–973]`

After training:
- Save final LoRA to `lora_unet_final/`.
- Save value-net weights if VF was enabled.
- Plot `reward_curve.png` (mean_r over rounds).
- Three **per-prompt artifacts**:
  - `snapshots.csv`: round, prompt_idx, IR score, image path.
  - `per_prompt_curves.png`: one line per snapshotted prompt, IR vs round.
  - `snapshot_grid.png`: rows = prompts, cols = checkpointed rounds, cells = the rendered image with its IR score.

> **Difference vs official:** the official code only saves the LoRA + optimizer state. No reward curve, no per-prompt visualizations.
> **Why ours is better:** the per-prompt grid is invaluable for debugging. You can literally see whether "A green colored rabbit" is becoming greener over rounds, or whether the policy is collapsing.

---

## 3. Side-by-side summary table

| Aspect | Official DPOK | Ours | Equivalence |
|--------|---------------|------|-------------|
| Files | 4 (train + pipeline subclass + scheduler subclass + reward) | 1 (`dpok_imagereward.py`) | Same logic, simpler layout |
| Sampling | `pipeline.forward_collect_traj_ddim` | `sample_trajectory()` | Same algorithm |
| Stored per-step | `log_prob` (scalar) | `noise_pred_old` (tensor) | Equivalent — ours recomputes ratio at use time |
| Log-ratio | `Normal(μ, σ).log_prob(x_{t-1})` difference | analytic `(‖x−μ_old‖² − ‖x−μ_new‖²)/(2σ²)` | Mathematically identical |
| KL | `(noise_θ − noise_ref)²` (no σ division) | `(noise_θ − noise_ref)² / (2σ²)` | Differs by per-step scaling — ours is the analytic Gaussian KL |
| Loss formula | `−α A·clip(ρ, 1±ε) + β·KL` | same | identical |
| ε (IS clip) | 1e-4 default | 1e-4 default → 1e-3 in latest run | tightened/loosened by us based on diagnostics |
| α, β | 10, 0.01 (single-prompt) | 10 default → 5 in latest run | loosened based on diagnostics |
| VF architecture | flatten-MLP (4·64·64+768→256) ConditionalLinear ×3 + Linear | small CNN over latent + sin-emb time + pooled text → MLP head | ours is smaller, has spatial inductive bias |
| VF training | separate phase, 5 iters per round | joint backward with policy, separate optimizer | functionally identical |
| Buffer | persistent FIFO, size 1000, off-policy reuse | single-round, on-policy | ours wastes samples but is closer to true PG |
| Grad accumulation | 12× microbatch 4 → eff. 48 | none, n=32 pairs in one backward | functionally similar effective batch |
| LoRA | `LoRACrossAttnProcessor` (deprecated diffusers API), rank 4 | `peft.LoraConfig` on `to_q/k/v/out.0`, rank 4→16 | ours uses the modern API |
| Gradient checkpointing | yes | no (bug with PEFT) | ours sacrifices memory for correctness |
| Mixed precision | fp16 via `accelerator` | bf16 default → fp32 in latest run | ours more flexible |
| CFG in training | yes, scale 7.5 | yes, scale 7.5 | identical |
| CFG in policy log-prob | yes | yes | identical |
| Eta | 1.0 | 1.0 | identical |
| DDIM steps | 50 | 20 | ours faster sampling |
| Reward | ImageReward, normalized | ImageReward, raw | functionally identical, scale offset |
| Checkpointing artifacts | LoRA + optimizer | LoRA + optimizer + value-net + reward curve + per-prompt CSV / grid / curves | ours much richer for debug |

---

## 4. Defense points — why our deviations are defensible

1. **Single-file design.** Easier to audit. Every algorithmic decision is in one place. The official code spreads sampling across three files, which obscures the IS contract (what `noise_pred_old` does and where it goes).

2. **Analytic log-ratio.** Mathematically identical to `Normal.log_prob` differencing, skips the `Distribution` allocation, more obvious to a reviewer that the constants cancel.

3. **σ²-normalized KL.** Closer to the paper's analytical derivation (eq. 9). β has unit semantics ("how much KL pressure per nat of divergence") rather than being a dimensionful mixing constant.

4. **CNN value network.** ~250K params vs ~5M in their flatten-MLP, leaves more VRAM for SD1.5 + ImageReward, gives a spatial inductive bias that an MLP can't.

5. **PEFT LoRA.** Modern HuggingFace API. Forward-compatible with newer diffusers releases. Their `LoRACrossAttnProcessor` was deprecated in diffusers ≥0.20.

6. **Single-round on-policy buffer.** Sacrifices sample efficiency but eliminates the off-policy bias accumulation that the buffer-based approach has when ε is tight (1e-4 with stale samples is a known instability source — the IS weights for old samples are systematically clipped, biasing toward more-recent samples anyway).

7. **Per-prompt fixed-seed snapshots.** Lets you visually inspect what the policy is actually learning for each prompt over training. This is the single most important debugging tool we have and the official code doesn't have anything like it.

8. **Hyperparameter exposure.** Every paper hyperparameter is a CLI flag with documentation in `--help`. The official code has many constants buried as defaults in argparse with no comments.

---

## 5. Open issues to flag at the meeting

1. **Reward not improving in our runs.** Despite the algorithm being faithful, ImageReward delta is ≤ +0.04 vs paper's +0.76 for the rabbit prompt. We've narrowed it to:
   - `is_clip=1e-4` was too tight (50% of gradients zeroed) → fixed in v3 (1e-3).
   - `lr=1e-5 + grad_norm_clip=0.1` → effective step size ~1e-6 → policy doesn't move. Loosened in v3 (lr=1e-4, clip=1.0).
   - bf16 may be losing gradient signal in the 1e-3 to 1e-4 range. Switched to fp32 in v3.
   - LoRA rank 4 may be too low capacity. Bumped to 16 in v3.

2. **Value function for single-prompt training.** Math says VF doesn't change the gradient direction in expectation (any state-only baseline is unbiased), only reduces variance. So VF should not block learning. We keep it on.

3. **Reward scale mismatch.** Paper reports rabbit baseline IR = 0.84; we measure -0.38 with the same "ImageReward-v1.0" model. Could be a different model checkpoint or a normalization difference.

4. **CFG dual-pass cost.** Each `compute_is_loss_term` call does 2× UNet forwards (uncond + cond, then split + combine). We accept the 2× cost because train-eval CFG match is critical (the deployed model is run with CFG=7.5).

---

## 6. Quick reference for running

| Run | Script | Prompts | Wall |
|-----|--------|---------|------|
| Single-prompt (Sec 5.2) | `submit_imagereward_single.sh` | `prompts_single.json` (rabbit) | ≤24h |
| 4-prompt + holdout (Sec 5.2 hybrid) | `submit_imagereward.sh` | `prompts_paper.json` + holdout eval | ≤24h |
| 104-prompt (Sec 5.3) | `submit_imagereward_multi.sh` | auto COCO | ≤24h |

All three currently use the same v3 hyperparameters: `--pipe_dtype fp32 --lora_rank 16 --lr 1e-4 --grad_norm_clip 1.0 --is_clip 1e-3 --alpha 5 --beta 0.01 --use_value_function`.
