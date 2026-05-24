"""
DPOK with ImageReward + paper-faithful importance sampling.

Branches from `dpok_hpc_wandb.py`, but faithful to Fan et al. 2023 (arXiv 2305.16381)
Appendix B (Online RL) and A.6 (importance sampling + PPO-style ratio clip):

  * Reward       : ImageReward (Xu et al. 2023) — replaces CLIP-ViT-B-32 cosine.
                   Trained on human preferences → wider dynamic range than CLIP
                   similarity (~[-3, +3] vs ~[0.2, 0.35]), better SNR for RL.
  * CFG          : guidance_scale=7.5 in BOTH training sampling AND the policy's
                   log-prob, matching inference/eval. No train-test mismatch.
  * IS           : store noise_pred_old from the sampling pass; reuse each
                   sampled trajectory for K gradient steps; weight by clipped
                   ratio  p_theta(x_{t-1}|x_t,z) / p_theta_old(x_{t-1}|x_t,z)
                   with epsilon = 1e-4 (paper eq. 26).
  * Pool         : all (x_t, x_{t-1}, t) pairs from the m sampled trajectories
                   are pooled; each gradient step samples n pairs with
                   replacement (paper Appendix B: m=10, n=32).
  * Hyperparams  : alpha=10, beta=0.01, lr=1e-5, grad-norm clip=0.1, K=5, n=32
                   — the paper's exact online RL setting.

NOT included (explicit choices): value-function baseline (paper A.5 — recommended
as follow-up if reward is flat; reduces variance), multi-prompt training-size
tweaks (paper E.3). LoRA rank, save/eval cadence, and W&B plumbing match the
existing CLIP script so the two runs are directly comparable.

Run (same CLI shape as dpok_hpc_wandb.py):
  python -u dpok_imagereward.py --coco ... --total_samples 750 \
      --sample_batch 10 --grad_steps 5 --is_batch 32 --is_clip 1e-4 \
      --guidance_scale 7.5 --pipe_dtype bf16 --save_dir ...
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from peft import LoraConfig, get_peft_model


# -----------------------------------------------------------------------------
# 0. Arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DPOK + ImageReward + paper-faithful IS")
# Data
parser.add_argument("--prompts_file",     type=str,   default=None,
                    help="JSON list of prompts (paper-style: 4 curated compositional prompts). "
                         "If set, --coco/--psg/--gqa are ignored.")
parser.add_argument("--coco",             type=str,   default=None)
parser.add_argument("--psg",              type=str,   default=None)
parser.add_argument("--gqa",              type=str,   default=None)
parser.add_argument("--max_prompts",      type=int,   default=500)
# Paper hyperparameters (Appendix B, Online RL training)
parser.add_argument("--total_samples",    type=int,   default=20000,
                    help="Total online samples (paper: 20000 ≈ 10K gradient steps)")
parser.add_argument("--sample_batch",     type=int,   default=10,
                    help="m in paper — trajectories sampled per round (paper Sec 5.2: m=10)")
parser.add_argument("--grad_steps",       type=int,   default=5,
                    help="K in paper — gradient updates per sample round using IS")
parser.add_argument("--is_batch",         type=int,   default=32,
                    help="n in paper — (x_t,x_{t-1}) pairs per gradient step")
parser.add_argument("--is_clip",          type=float, default=1e-4,
                    help="epsilon in paper eq. 26 — PPO-style IS ratio clip")
parser.add_argument("--num_steps",        type=int,   default=20,
                    help="DDIM sampling steps")
parser.add_argument("--alpha",            type=float, default=10.0,
                    help="policy reward weight (paper Sec 5.2 text: α=10)")
parser.add_argument("--beta",             type=float, default=0.01,
                    help="KL weight (paper: 0.01)")
parser.add_argument("--lr",               type=float, default=1e-5,
                    help="AdamW learning rate (paper: 1e-5)")
parser.add_argument("--grad_norm_clip",   type=float, default=0.1,
                    help="clip on grad norm (paper: < 0.1)")
parser.add_argument("--lora_rank",        type=int,   default=4)
parser.add_argument("--guidance_scale",   type=float, default=7.5,
                    help="CFG scale used in sampling AND in policy log-prob. "
                         "Default 7.5 matches eval. Set 1.0 to disable CFG.")
# IO / runtime
parser.add_argument("--save_dir",         type=str,   default="dpok_outputs")
parser.add_argument("--save_every",       type=int,   default=25)
parser.add_argument("--wandb_project",    type=str,   default="dpok-imagereward")
parser.add_argument("--wandb_name",       type=str,   default=None)
parser.add_argument("--pipe_dtype",       type=str,   default="bf16",
                    choices=["auto", "fp32", "fp16", "bf16"])
parser.add_argument("--seed",             type=int,   default=42)
# Numerics
parser.add_argument("--eta",              type=float, default=1.0,
                    help="DDIM eta — 1.0 makes the policy stochastic (paper uses DDPM "
                         "which is equivalent to DDIM eta=1).")
parser.add_argument("--sigma_floor",      type=float, default=1e-5)
parser.add_argument("--sample_retries",   type=int,   default=3)
parser.add_argument("--reward_clip_abs",  type=float, default=5.0,
                    help="abs-clip the per-sample reward before policy loss; ImageReward is "
                         "roughly [-3, +3] but occasional outliers reach ±6.")
# Value function (paper Appendix A.5) — variance reduction baseline
parser.add_argument("--use_value_function", action="store_true",
                    help="enable V(x_t,z,θ) baseline; advantage = (r - V) replaces r in policy loss")
parser.add_argument("--vf_lr",            type=float, default=1e-4,
                    help="AdamW LR for value network (paper default: 1e-4)")
parser.add_argument("--vf_weight",        type=float, default=0.5,
                    help="MSE weight on value-function loss; total = policy + KL + vf_weight * (V-r)^2")
args = parser.parse_args()


# -----------------------------------------------------------------------------
# 1. Setup
# -----------------------------------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

ALPHA            = args.alpha
BETA             = args.beta
LR               = args.lr
M_SAMPLE_BATCH   = args.sample_batch
K_GRAD_STEPS     = args.grad_steps
N_IS_BATCH       = args.is_batch
IS_CLIP          = args.is_clip
TOTAL_SAMPLES    = args.total_samples
NUM_STEPS        = args.num_steps
GRAD_NORM_CLIP   = args.grad_norm_clip
LORA_RANK        = args.lora_rank
ETA              = args.eta
SIGMA_FLOOR      = args.sigma_floor
GUIDANCE_SCALE   = args.guidance_scale
DO_CFG           = GUIDANCE_SCALE > 1.0 + 1e-6
SAMPLE_RETRIES   = max(1, args.sample_retries)
REWARD_CLIP_ABS  = args.reward_clip_abs
SAVE_EVERY       = args.save_every
USE_VF           = args.use_value_function
VF_LR            = args.vf_lr
VF_WEIGHT        = args.vf_weight

SAVE_DIR = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "checkpoints").mkdir(exist_ok=True)
(SAVE_DIR / "samples").mkdir(exist_ok=True)
(SAVE_DIR / "inference").mkdir(exist_ok=True)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"


def choose_dtype(name: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    if name == "fp32": return torch.float32
    if name == "fp16": return torch.float16
    if name == "bf16": return torch.bfloat16
    if torch.cuda.is_bf16_supported(): return torch.bfloat16
    return torch.float16


model_dtype = choose_dtype(args.pipe_dtype)
print(f"Device: {device}  dtype: {model_dtype}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def zero_loss() -> torch.Tensor:
    return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)


# -----------------------------------------------------------------------------
# 2. W&B
# -----------------------------------------------------------------------------
run = wandb.init(
    project=args.wandb_project,
    name=args.wandb_name,
    config={
        "algorithm": "DPOK + ImageReward + IS (paper-faithful)",
        "alpha": ALPHA, "beta": BETA, "lr": LR,
        "m_sample_batch": M_SAMPLE_BATCH,
        "k_grad_steps_per_round": K_GRAD_STEPS,
        "n_is_batch": N_IS_BATCH,
        "is_clip_epsilon": IS_CLIP,
        "total_samples": TOTAL_SAMPLES,
        "num_steps": NUM_STEPS,
        "grad_norm_clip": GRAD_NORM_CLIP,
        "lora_rank": LORA_RANK,
        "eta": ETA,
        "sigma_floor": SIGMA_FLOOR,
        "pipe_dtype": str(model_dtype),
        "base_model": "runwayml/stable-diffusion-v1-5",
        "lora_targets": "to_q,to_k,to_v,to_out.0",
        "reward_model": "ImageReward-v1.0",
        "guidance_scale": GUIDANCE_SCALE,
        "do_cfg_in_training": DO_CFG,
        "use_value_function": USE_VF,
        "vf_lr": VF_LR,
        "vf_weight": VF_WEIGHT,
        "data_coco": args.coco,
        "data_psg": args.psg,
        "data_gqa": args.gqa,
        "max_prompts": args.max_prompts,
    },
    save_code=False,
    dir=str(SAVE_DIR),
)
wandb.define_metric("train/*",    step_metric="update_step")
wandb.define_metric("reward/*",   step_metric="update_step")
wandb.define_metric("round/*",    step_metric="update_step")
wandb.define_metric("is/*",       step_metric="update_step")
wandb.define_metric("loss/*",     step_metric="update_step")
wandb.define_metric("vf/*",       step_metric="update_step")
wandb.define_metric("gpu/*",      step_metric="update_step")

print(f"W&B run: {run.url}")


# -----------------------------------------------------------------------------
# 3. Pipeline + reference UNet
# -----------------------------------------------------------------------------
print("Loading SD1.5 ...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=model_dtype,
    safety_checker=None,
).to(device)
pipe.set_progress_bar_config(disable=True)

pipe_ref_unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    torch_dtype=model_dtype,
).to(device)
for p in pipe_ref_unet.parameters():
    p.requires_grad_(False)
pipe_ref_unet.eval()

# VAE in fp32 for decode stability.
pipe.vae.to(device=device, dtype=torch.float32)
print("Models loaded.")


# -----------------------------------------------------------------------------
# 4. Dataset
# -----------------------------------------------------------------------------
print("Loading prompts ...")
if args.prompts_file:
    import json as _json
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        all_prompts = _json.load(f)
    if not isinstance(all_prompts, list) or not all(isinstance(p, str) for p in all_prompts):
        raise ValueError(f"--prompts_file must be a JSON list of strings: {args.prompts_file}")
elif args.coco or args.psg or args.gqa:
    from dpok_scene_reward import load_coco_items, load_psg_items, load_gqa_items
    all_items = []
    if args.coco:
        all_items += load_coco_items(args.coco, max_captions=args.max_prompts, seed=args.seed)
    if args.psg:
        all_items += load_psg_items(args.psg, max_scenes=args.max_prompts)
    if args.gqa:
        all_items += load_gqa_items(args.gqa, max_scenes=args.max_prompts)
    random.shuffle(all_items)
    all_prompts = [item.text for item in all_items]
else:
    raise ValueError("Pass --prompts_file or --coco/--psg/--gqa.")
print(f"Loaded {len(all_prompts)} prompts. Example: {all_prompts[0]}")


# -----------------------------------------------------------------------------
# 5. LoRA
# -----------------------------------------------------------------------------
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none",
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.print_trainable_parameters()
pipe.unet.to(device=device, dtype=model_dtype)
# (No gradient checkpointing — breaks lora_B grad, see dpok_hpc_wandb.py note.)


# -----------------------------------------------------------------------------
# 6. Optimizer
# -----------------------------------------------------------------------------
trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR)
print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")


# -----------------------------------------------------------------------------
# 6b. Value network V(x_t, t, z) — paper Appendix A.5
# -----------------------------------------------------------------------------
# Small CNN over the 4×64×64 latent + sinusoidal time embedding + pooled prompt
# embedding → scalar V. Trained with MSE against the (clipped) terminal reward.
# Advantage = (r - V.detach()) replaces r in the policy loss → variance reduction
# without biasing the gradient (V's grad path is severed via .detach()).
class ValueNetwork(torch.nn.Module):
    def __init__(self, latent_ch: int = 4, time_dim: int = 128,
                 prompt_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(latent_ch, 32, 3, stride=2, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),         torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),        torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_dim, hidden), torch.nn.SiLU(),
        )
        self.prompt_mlp = torch.nn.Sequential(
            torch.nn.Linear(prompt_dim, hidden), torch.nn.SiLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128 + hidden + hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def _sinusoidal_time_emb(self, t_val: int, device, dtype) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        ang = float(t_val) * freqs
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (time_dim,)
        return emb.to(dtype=dtype).unsqueeze(0)                     # (1, time_dim)

    def forward(self, x_t: torch.Tensor, t_val: int, prompt_embeds: torch.Tensor) -> torch.Tensor:
        # x_t: (1, 4, H, W) fp32, prompt_embeds: (1, 77, 768)
        h_x = self.conv(x_t)                                                        # (1, 128)
        h_t = self.time_mlp(self._sinusoidal_time_emb(t_val, x_t.device, x_t.dtype))# (1, hidden)
        h_z = self.prompt_mlp(prompt_embeds.mean(dim=1).to(x_t.dtype))              # (1, hidden)
        h = torch.cat([h_x, h_t, h_z], dim=-1)
        return self.head(h).squeeze(-1)                                             # (1,)


value_net = None
vf_optimizer = None
if USE_VF:
    value_net = ValueNetwork().to(device=device, dtype=torch.float32)
    vf_optimizer = torch.optim.AdamW(value_net.parameters(), lr=VF_LR)
    n_vf = sum(p.numel() for p in value_net.parameters())
    print(f"Value network enabled. Params: {n_vf:,}  vf_lr={VF_LR}  vf_weight={VF_WEIGHT}")
else:
    print("Value network: DISABLED (use --use_value_function to enable)")


# -----------------------------------------------------------------------------
# 7. Reward — ImageReward
# -----------------------------------------------------------------------------
# The `image-reward` PyPI package. First use downloads ~1.5GB of weights.
# If HF_HUB_OFFLINE=1, BERT-base-uncased must already be cached (it is on our
# cluster — used by eval_report.py's ImageReward metric).
print("Loading ImageReward ...")
try:
    import ImageReward as RM
    _ir_model = RM.load("ImageReward-v1.0", device=device)
    _ir_model.eval()
    for p in _ir_model.parameters():
        p.requires_grad_(False)
except Exception as e:
    raise RuntimeError(
        "Failed to load ImageReward. On this HPC, make sure\n"
        "  pip install image-reward\n"
        "is done in the dpok env AND that BERT-base-uncased is cached under\n"
        "HF_HOME. If HF_HUB_OFFLINE=1, run one eval_report.py job first to\n"
        "populate the cache, then retry training.\n"
        f"Original error: {e}"
    )


@torch.no_grad()
def compute_reward(pil_image: Image.Image, prompt: str) -> torch.Tensor:
    """Returns a scalar fp32 tensor on `device` ≈ ImageReward score."""
    score = _ir_model.score(prompt, pil_image.convert("RGB"))
    return torch.tensor(float(score), device=device, dtype=torch.float32)


blank_score = compute_reward(Image.new("RGB", (64, 64)), "a cat")
print(f"ImageReward ready. Blank-image baseline score: {blank_score.item():+.4f}")


# -----------------------------------------------------------------------------
# 8. Scheduler utilities
# -----------------------------------------------------------------------------
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
scheduler = pipe.scheduler
scheduler.set_timesteps(NUM_STEPS, device=device)

timestep_list = scheduler.timesteps.tolist()
next_denoising_step = {}
for i, t_val in enumerate(timestep_list):
    next_denoising_step[int(t_val)] = int(timestep_list[i + 1]) if i + 1 < len(timestep_list) else -1


@torch.no_grad()
def get_sigma_sq(t_val: int) -> torch.Tensor:
    next_t = next_denoising_step[t_val]
    ac_t = scheduler.alphas_cumprod[t_val].float().to(device)
    ac_next = scheduler.alphas_cumprod[next_t].float().to(device) if next_t >= 0 else torch.tensor(1.0, device=device)
    beta_t = 1.0 - ac_t / ac_next
    sigma_sq = ((1.0 - ac_next) / (1.0 - ac_t)) * beta_t
    sigma_sq = (ETA ** 2) * sigma_sq
    return sigma_sq.clamp(min=SIGMA_FLOOR)


def noise_pred_to_mu(x_t: torch.Tensor, noise_pred: torch.Tensor, t_val: int) -> torch.Tensor:
    """Linear map noise_pred → mu_theta. MUST propagate gradient."""
    next_t = next_denoising_step[t_val]
    ac_t = scheduler.alphas_cumprod[t_val].to(dtype=x_t.dtype, device=x_t.device)
    ac_next = scheduler.alphas_cumprod[next_t].to(dtype=x_t.dtype, device=x_t.device) \
              if next_t >= 0 else torch.tensor(1.0, dtype=x_t.dtype, device=x_t.device)
    alpha_t = ac_t / ac_next
    beta_t = 1.0 - alpha_t
    mu = (1.0 / alpha_t.sqrt()) * (x_t - beta_t / (1.0 - ac_t).sqrt() * noise_pred)
    return mu


# -----------------------------------------------------------------------------
# 9. Sampling
# -----------------------------------------------------------------------------
@torch.no_grad()
def decode_latents_to_pil(latents: torch.Tensor) -> Image.Image:
    latents_fp32 = latents.to(device=device, dtype=torch.float32)
    if not is_finite_tensor(latents_fp32):
        raise FloatingPointError("Non-finite latents before VAE decode")
    image_tensor = pipe.vae.decode(
        latents_fp32 / pipe.vae.config.scaling_factor,
        return_dict=False,
    )[0].float()
    if not is_finite_tensor(image_tensor):
        raise FloatingPointError("Non-finite decoded image tensor")
    image_tensor = ((image_tensor / 2.0) + 0.5).clamp(0.0, 1.0)
    mean_val, std_val = float(image_tensor.mean().item()), float(image_tensor.std().item())
    if mean_val < 1e-4 and std_val < 1e-4:
        raise FloatingPointError("Collapsed near-black image")
    image_np = (image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_np)


@torch.no_grad()
def sample_trajectory(prompt_embeds: torch.Tensor, uncond_embeds: torch.Tensor):
    """
    Returns:
        steps = [(x_t, x_t_minus_1, t_val, noise_pred_old), ...]   (all detached fp32)
        image_pil                                                  (final decode)

    noise_pred_old is the CFG-guided noise prediction that was actually used to
    produce x_{t-1} (the one that parameterized p_theta_old for IS).
    """
    prompt_embeds_model = prompt_embeds.to(device=device, dtype=model_dtype)
    use_cfg = DO_CFG and uncond_embeds is not None
    if use_cfg:
        uncond_embeds_model = uncond_embeds.to(device=device, dtype=model_dtype)
        embeds_cat = torch.cat([uncond_embeds_model, prompt_embeds_model], dim=0)

    h = w = pipe.unet.config.sample_size
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, h, w),
        device=device, dtype=model_dtype,
    )

    steps = []
    for t in scheduler.timesteps:
        t_val = int(t.item())
        x_t = latents.detach().float().clone()

        if use_cfg:
            latent_in = scheduler.scale_model_input(torch.cat([latents, latents], dim=0), t)
            both = pipe.unet(latent_in, t, encoder_hidden_states=embeds_cat).sample
            nu, nc = both.chunk(2, dim=0)
            noise_pred = nu + GUIDANCE_SCALE * (nc - nu)
        else:
            latent_in = scheduler.scale_model_input(latents, t)
            noise_pred = pipe.unet(latent_in, t, encoder_hidden_states=prompt_embeds_model).sample

        if not is_finite_tensor(noise_pred.float()):
            raise FloatingPointError(f"Non-finite noise_pred at t={t_val}")
        noise_pred_old = noise_pred.detach().float().clone()

        out = scheduler.step(noise_pred, t, latents, eta=ETA)
        latents = out.prev_sample
        if not is_finite_tensor(latents.float()):
            raise FloatingPointError(f"Non-finite latents at t={t_val}")

        x_t_minus_1 = latents.detach().float().clone()
        steps.append((x_t, x_t_minus_1, t_val, noise_pred_old))

    image_pil = decode_latents_to_pil(latents)
    return steps, image_pil


@torch.no_grad()
def safe_inference_image(prompt: str, num_inference_steps: int = 50,
                         guidance_scale: float = 7.5,
                         generator: torch.Generator = None) -> Image.Image:
    out = pipe(prompt, num_inference_steps=num_inference_steps,
               guidance_scale=guidance_scale, output_type="latent",
               generator=generator)
    latents = out.images
    if isinstance(latents, (list, tuple)):
        latents = latents[0]
    return decode_latents_to_pil(latents)


# -----------------------------------------------------------------------------
# 10. IS loss over a single (x_t, x_{t-1}, t, noise_pred_old) pair
# -----------------------------------------------------------------------------
def compute_is_loss_term(x_t: torch.Tensor,
                         x_t_minus_1: torch.Tensor,
                         t_val: int,
                         noise_pred_old: torch.Tensor,
                         reward: torch.Tensor,
                         prompt_embeds: torch.Tensor,
                         uncond_embeds: torch.Tensor):
    """
    Paper-faithful (eq. 9 + eq. 26):
        policy term = - alpha * r * clip( p_theta / p_theta_old , 1-eps, 1+eps )
        KL term     = + beta  *          || noise_theta - noise_ref ||^2 / (2 sigma^2)

    p_theta is Gaussian with mean mu_theta(x_t, noise_theta_guided) and fixed
    variance sigma^2; log-prob = -||x_{t-1} - mu||^2 / (2 sigma^2) + const. The
    const cancels in the ratio, so:
        log_ratio = ( ||x_{t-1} - mu_old||^2 - ||x_{t-1} - mu_new||^2 ) / (2 sigma^2)

    Returns (loss_scalar, stats_dict) both as fp32. stats are floats for W&B.
    """
    x_t = x_t.to(device=device, dtype=torch.float32)
    x_t_minus_1 = x_t_minus_1.to(device=device, dtype=torch.float32).detach()
    noise_pred_old = noise_pred_old.to(device=device, dtype=torch.float32).detach()

    sigma_sq = get_sigma_sq(t_val).to(torch.float32)
    t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
    latent_in = scheduler.scale_model_input(x_t, t_tensor)

    prompt_embeds_model = prompt_embeds.to(device=device, dtype=model_dtype)
    use_cfg = DO_CFG and uncond_embeds is not None
    if use_cfg:
        uncond_embeds_model = uncond_embeds.to(device=device, dtype=model_dtype)
        embeds_cat = torch.cat([uncond_embeds_model, prompt_embeds_model], dim=0)
        latent_in_cat = torch.cat([latent_in, latent_in], dim=0).to(dtype=model_dtype)
        both_tr = pipe.unet(latent_in_cat, t_tensor, encoder_hidden_states=embeds_cat).sample.to(torch.float32)
        nu_tr, nc_tr = both_tr.chunk(2, dim=0)
        noise_tr = nu_tr + GUIDANCE_SCALE * (nc_tr - nu_tr)
        with torch.no_grad():
            both_ref = pipe_ref_unet(latent_in_cat, t_tensor, encoder_hidden_states=embeds_cat).sample.to(torch.float32)
            nu_rf, nc_rf = both_ref.chunk(2, dim=0)
            noise_ref = nu_rf + GUIDANCE_SCALE * (nc_rf - nu_rf)
    else:
        noise_tr = pipe.unet(latent_in.to(dtype=model_dtype), t_tensor,
                             encoder_hidden_states=prompt_embeds_model).sample.to(torch.float32)
        with torch.no_grad():
            noise_ref = pipe_ref_unet(latent_in.to(dtype=model_dtype), t_tensor,
                                      encoder_hidden_states=prompt_embeds_model).sample.to(torch.float32)

    if not is_finite_tensor(noise_tr) or not is_finite_tensor(noise_ref):
        return None, None

    mu_new = noise_pred_to_mu(x_t, noise_tr, t_val)
    mu_old = noise_pred_to_mu(x_t, noise_pred_old, t_val).detach()
    if not is_finite_tensor(mu_new):
        return None, None

    diff_new_sq = ((x_t_minus_1 - mu_new) ** 2).mean()
    diff_old_sq = ((x_t_minus_1 - mu_old) ** 2).mean()
    log_ratio = (diff_old_sq - diff_new_sq) / (2.0 * sigma_sq)
    # Numerical guard — at init (theta == theta_old) log_ratio is ~0.
    log_ratio = log_ratio.clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    ratio_clipped = torch.clip(ratio, 1.0 - IS_CLIP, 1.0 + IS_CLIP)

    # KL term: gradient flows through noise_tr only (noise_ref is detached).
    kl_num = ((noise_tr - noise_ref.detach()) ** 2).mean()
    kl_t = kl_num / (2.0 * sigma_sq)

    r_clipped = reward.detach().to(device=device, dtype=torch.float32).clamp(
        min=-REWARD_CLIP_ABS, max=REWARD_CLIP_ABS
    )

    # ── Value function (paper A.5): advantage = r - V(x_t,t,z) ──────────────
    if USE_VF and value_net is not None:
        v_pred = value_net(x_t, t_val, prompt_embeds.to(device=device, dtype=torch.float32))
        v_pred = v_pred.squeeze()  # scalar
        advantage = (r_clipped - v_pred.detach())
        # MSE regression target = the realized terminal reward.
        vf_loss = VF_WEIGHT * (v_pred - r_clipped) ** 2
    else:
        v_pred = torch.zeros((), device=device, dtype=torch.float32)
        advantage = r_clipped
        vf_loss = torch.zeros((), device=device, dtype=torch.float32)

    policy_loss = -ALPHA * advantage * ratio_clipped
    kl_loss = BETA * kl_t
    loss = policy_loss + kl_loss + vf_loss

    if not is_finite_tensor(loss):
        return None, None

    stats = {
        "ratio": float(ratio.detach()),
        "ratio_clipped": float(ratio_clipped.detach()),
        "log_ratio": float(log_ratio.detach()),
        "was_clipped": bool((ratio.detach() < 1 - IS_CLIP) or (ratio.detach() > 1 + IS_CLIP)),
        "policy_loss": float(policy_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "vf_loss": float(vf_loss.detach()),
        "v_pred": float(v_pred.detach()),
        "advantage": float(advantage.detach()),
        "reward": float(r_clipped.detach()),
    }
    return loss, stats


# -----------------------------------------------------------------------------
# 11. Training loop
# -----------------------------------------------------------------------------
samples_generated = 0
update_step = 0
reward_history = []
loss_history = []
skipped_rounds = 0

# Per-prompt snapshot tracking. snapshots[prompt_idx] = list of (round, ir_score, img_path).
# Same seed per prompt across all rounds, so the only thing changing is the LoRA.
# Cap at 8 prompts to keep snapshot overhead bounded for the 104-prompt regime.
SNAPSHOT_SEED_BASE = 42
SNAPSHOT_MAX_PROMPTS = min(8, len(all_prompts))
snapshot_prompt_idxs = list(range(SNAPSHOT_MAX_PROMPTS))
snapshots = {i: [] for i in snapshot_prompt_idxs}
# Save the prompt list so we know which index is which prompt later.
import json as _json_save
with open(SAVE_DIR / "snapshot_prompts.json", "w", encoding="utf-8") as _f:
    _json_save.dump(all_prompts, _f, indent=2)

print(f"\nDPOK-IR training: target {TOTAL_SAMPLES} online samples")
print(f"  m={M_SAMPLE_BATCH}  K={K_GRAD_STEPS}  n={N_IS_BATCH}  eps={IS_CLIP}")
print(f"  alpha={ALPHA}  beta={BETA}  lr={LR}  grad_clip={GRAD_NORM_CLIP}")
print(f"  guidance_scale={GUIDANCE_SCALE}  do_cfg_in_training={DO_CFG}")
print("-" * 60)

while samples_generated < TOTAL_SAMPLES:
    # ── Sampling phase ────────────────────────────────────────────────────
    pipe.unet.eval()
    buffer = []          # list of (steps, reward, prompt_embeds, uncond_embeds, prompt)
    raw_rewards = []

    for _ in range(M_SAMPLE_BATCH):
        prompt = random.choice(all_prompts)
        for attempt in range(SAMPLE_RETRIES):
            try:
                with torch.no_grad():
                    prompt_embeds, uncond_embeds = pipe.encode_prompt(
                        prompt=prompt, device=device, num_images_per_prompt=1,
                        do_classifier_free_guidance=DO_CFG,
                    )
                    prompt_embeds = prompt_embeds.detach().to(torch.float32)
                    if DO_CFG and uncond_embeds is not None:
                        uncond_embeds = uncond_embeds.detach().to(torch.float32)
                    else:
                        uncond_embeds = None

                steps, image_pil = sample_trajectory(prompt_embeds, uncond_embeds)
                r = compute_reward(image_pil, prompt)
                raw_rewards.append(float(r.item()))
                buffer.append((steps, r, prompt_embeds, uncond_embeds, prompt))
                samples_generated += 1
                break
            except Exception as exc:
                if attempt + 1 == SAMPLE_RETRIES:
                    print(f"WARNING: sample skipped after {SAMPLE_RETRIES} tries: {exc}")
                continue

    torch.cuda.empty_cache()

    if len(buffer) == 0:
        print("WARNING: empty sample buffer, continuing")
        continue

    # ── Build (x_t, x_{t-1}, t, noise_pred_old, r, pe, ue) pool ───────────
    pool = []
    for (steps, r, pe, ue, _prompt) in buffer:
        for (x_t, x_tm1, t_val, npold) in steps:
            if next_denoising_step[t_val] >= 0:
                pool.append((x_t, x_tm1, t_val, npold, r, pe, ue))
    if len(pool) == 0:
        print("WARNING: empty step pool, continuing")
        continue

    # ── K gradient steps with IS sampling of n pairs each ─────────────────
    pipe.unet.train()
    round_policy_loss = 0.0
    round_kl_loss = 0.0
    round_vf_loss = 0.0
    round_v_vals = []
    round_adv_vals = []
    round_valid_terms = 0
    round_ratio_vals = []
    round_clip_hits = 0
    round_optimizer_steps = 0

    for grad_step in range(K_GRAD_STEPS):
        optimizer.zero_grad(set_to_none=True)
        if USE_VF and vf_optimizer is not None:
            vf_optimizer.zero_grad(set_to_none=True)

        # Sample n pairs with replacement (paper Appendix B).
        n = min(N_IS_BATCH, len(pool))
        picks = random.choices(pool, k=n)

        total_loss = zero_loss()
        valid_in_step = 0

        for (x_t, x_tm1, t_val, npold, r, pe, ue) in picks:
            loss_term, stats = compute_is_loss_term(x_t, x_tm1, t_val, npold, r, pe, ue)
            if loss_term is None:
                continue
            total_loss = total_loss + loss_term / n
            valid_in_step += 1
            round_policy_loss += stats["policy_loss"] / n
            round_kl_loss += stats["kl_loss"] / n
            round_vf_loss += stats.get("vf_loss", 0.0) / n
            round_v_vals.append(stats.get("v_pred", 0.0))
            round_adv_vals.append(stats.get("advantage", 0.0))
            round_ratio_vals.append(stats["ratio"])
            round_clip_hits += int(stats["was_clipped"])
            round_valid_terms += 1

        if valid_in_step == 0 or not total_loss.requires_grad:
            optimizer.zero_grad(set_to_none=True)
            if USE_VF and vf_optimizer is not None:
                vf_optimizer.zero_grad(set_to_none=True)
            skipped_rounds += 1
            continue

        total_loss.backward()

        # ── DIAG: confirm lora_B grad is non-zero on the very first step ───
        if update_step == 0 and grad_step == 0:
            b_with_grad = 0
            b_grad_norms = []
            for n_param, p in pipe.unet.named_parameters():
                if "lora_B" in n_param and p.requires_grad and p.grad is not None:
                    b_with_grad += 1
                    b_grad_norms.append(p.grad.float().norm().item())
            if b_grad_norms:
                arr = np.array(b_grad_norms)
                print(f"[DIAG] lora_B grads — n_nonzero={(arr > 0).sum()}/{len(arr)}  "
                      f"min={arr.min():.2e} max={arr.max():.2e} mean={arr.mean():.2e}",
                      flush=True)

        # Finite-grad guard.
        bad = False
        for p in trainable_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad = True; break
        if bad:
            optimizer.zero_grad(set_to_none=True)
            if USE_VF and vf_optimizer is not None:
                vf_optimizer.zero_grad(set_to_none=True)
            skipped_rounds += 1
            continue

        total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_NORM_CLIP)
        optimizer.step()
        if USE_VF and vf_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            vf_optimizer.step()
        update_step += 1
        round_optimizer_steps += 1

    # ── Round logging ──────────────────────────────────────────────────────
    mean_r = float(np.mean(raw_rewards)) if raw_rewards else 0.0
    reward_history.append(mean_r)
    if round_valid_terms > 0:
        loss_history.append(round_policy_loss + round_kl_loss)

    denom = max(1, round_valid_terms)
    wandb.log({
        "round/samples_generated": samples_generated,
        "round/mean_r":            mean_r,
        "round/min_r":             float(np.min(raw_rewards)) if raw_rewards else 0.0,
        "round/max_r":             float(np.max(raw_rewards)) if raw_rewards else 0.0,
        "loss/policy":             round_policy_loss / max(1, K_GRAD_STEPS),
        "loss/kl":                 round_kl_loss / max(1, K_GRAD_STEPS),
        "loss/total":               (round_policy_loss + round_kl_loss) / max(1, K_GRAD_STEPS),
        "is/ratio_mean":           float(np.mean(round_ratio_vals)) if round_ratio_vals else 1.0,
        "is/ratio_max":            float(np.max(round_ratio_vals)) if round_ratio_vals else 1.0,
        "is/ratio_min":            float(np.min(round_ratio_vals)) if round_ratio_vals else 1.0,
        "is/clip_hit_rate":        round_clip_hits / max(1, round_valid_terms),
        "train/optimizer_steps":   round_optimizer_steps,
        "train/skipped_rounds":    skipped_rounds,
        "vf/loss":                 round_vf_loss / max(1, K_GRAD_STEPS),
        "vf/v_mean":               float(np.mean(round_v_vals)) if round_v_vals else 0.0,
        "vf/advantage_mean":       float(np.mean(round_adv_vals)) if round_adv_vals else 0.0,
        "vf/advantage_std":        float(np.std(round_adv_vals)) if round_adv_vals else 0.0,
        "update_step":             update_step,
    }, step=update_step)

    if device == "cuda":
        wandb.log({
            "gpu/mem_alloc_gb":    torch.cuda.memory_allocated() / 1e9,
            "gpu/mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "update_step": update_step,
        }, step=update_step)

    print(f"[round {len(reward_history):4d}  samp {samples_generated:5d}/{TOTAL_SAMPLES}  "
          f"upd {update_step:5d}]  mean_r={mean_r:+.3f}  "
          f"pL={round_policy_loss/max(1,K_GRAD_STEPS):+.3e}  "
          f"kL={round_kl_loss/max(1,K_GRAD_STEPS):+.3e}  "
          f"ratio={np.mean(round_ratio_vals) if round_ratio_vals else 1.0:+.4f}  "
          f"clip={round_clip_hits}/{round_valid_terms}", flush=True)

    # ── Periodic checkpoint + inference image ─────────────────────────────
    if len(reward_history) % SAVE_EVERY == 0:
        ckpt_dir = SAVE_DIR / "checkpoints" / f"round_{len(reward_history):05d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        pipe.unet.save_pretrained(ckpt_dir / "lora_unet")
        state_blob = {
            "samples_generated": samples_generated,
            "update_step": update_step,
            "reward_history": reward_history,
            "loss_history": loss_history,
            "optimizer_state": optimizer.state_dict(),
        }
        if USE_VF and value_net is not None:
            state_blob["value_net_state"] = value_net.state_dict()
            state_blob["vf_optimizer_state"] = vf_optimizer.state_dict()
        torch.save(state_blob, ckpt_dir / "state.pt")
        print(f"  checkpoint: {ckpt_dir}", flush=True)

        # ── Per-prompt snapshot: render every prompt with FIXED per-prompt seed.
        # Same seed every round → the only thing that changes is the LoRA, so you
        # can flip through round_00025/prompt_00.png → round_00050/prompt_00.png →
        # … and watch one prompt evolve.
        try:
            pipe.unet.eval()
            round_dir = SAVE_DIR / "inference" / f"round_{len(reward_history):05d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            wandb_log = {"update_step": update_step}
            for p_idx in snapshot_prompt_idxs:
                p_text = all_prompts[p_idx]
                gen = torch.Generator(device=device).manual_seed(SNAPSHOT_SEED_BASE + p_idx)
                img = safe_inference_image(p_text, num_inference_steps=NUM_STEPS,
                                           guidance_scale=GUIDANCE_SCALE, generator=gen)
                img_path = round_dir / f"prompt_{p_idx:02d}.png"
                img.save(img_path)
                ir = float(compute_reward(img, p_text).item())
                snapshots[p_idx].append((len(reward_history), ir, str(img_path)))
                wandb_log[f"snapshot/ir_prompt_{p_idx}"] = ir
                wandb_log[f"snapshot/img_prompt_{p_idx}"] = wandb.Image(str(img_path), caption=p_text)
            wandb.log(wandb_log, step=update_step)
        except Exception as e:
            print(f"  snapshot failed: {e}")


# -----------------------------------------------------------------------------
# 12. Final save
# -----------------------------------------------------------------------------
print("\nTraining complete. Saving final LoRA ...")
final_dir = SAVE_DIR / "lora_unet_final"
pipe.unet.save_pretrained(final_dir)
print(f"Saved: {final_dir}")
if USE_VF and value_net is not None:
    torch.save(value_net.state_dict(), SAVE_DIR / "value_net_final.pt")
    print(f"Saved value net: {SAVE_DIR / 'value_net_final.pt'}")

# Learning curve
if reward_history:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(reward_history, label="mean_r per round")
    ax.set_xlabel("round")
    ax.set_ylabel("ImageReward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    curve_path = SAVE_DIR / "reward_curve.png"
    fig.tight_layout()
    fig.savefig(curve_path, dpi=100)
    plt.close(fig)
    wandb.log({"samples/reward_curve": wandb.Image(str(curve_path))})
    print(f"Learning curve: {curve_path}")


# ── Per-prompt snapshot artifacts ────────────────────────────────────────────
# Three outputs:
#   1. snapshots.csv         — round, prompt_idx, ir_score, img_path  (raw data)
#   2. per_prompt_curves.png — IR vs round, one line per prompt       (signal)
#   3. snapshot_grid.png     — N_PROMPTS rows × N_ROUNDS columns       (visual)
import csv
have_snapshots = any(len(v) > 0 for v in snapshots.values())
if have_snapshots:
    # 1. CSV
    csv_path = SAVE_DIR / "snapshots.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "prompt_idx", "prompt_text", "ir_score", "img_path"])
        for p_idx, rows in snapshots.items():
            for (rd, ir, ip) in rows:
                w.writerow([rd, p_idx, all_prompts[p_idx], ir, ip])
    print(f"Snapshots CSV: {csv_path}")

    # 2. Per-prompt learning curves
    fig, ax = plt.subplots(figsize=(9, 5))
    for p_idx, rows in snapshots.items():
        if not rows:
            continue
        rds = [r[0] for r in rows]
        irs = [r[1] for r in rows]
        ax.plot(rds, irs, marker="o", label=f"[{p_idx}] {all_prompts[p_idx][:50]}")
    ax.set_xlabel("round")
    ax.set_ylabel("ImageReward (fixed seed per prompt)")
    ax.set_title("Per-prompt training trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    pp_path = SAVE_DIR / "per_prompt_curves.png"
    fig.tight_layout()
    fig.savefig(pp_path, dpi=100)
    plt.close(fig)
    wandb.log({"samples/per_prompt_curves": wandb.Image(str(pp_path))})
    print(f"Per-prompt curves: {pp_path}")

    # 3. Snapshot grid: rows = prompts, cols = rounds  (only the snapshotted prompts)
    n_prompts = len(snapshot_prompt_idxs)
    rounds_present = sorted({r[0] for rows in snapshots.values() for r in rows})
    if rounds_present:
        n_cols = len(rounds_present)
        fig, axes = plt.subplots(n_prompts, n_cols,
                                 figsize=(2.0 * n_cols, 2.0 * n_prompts),
                                 squeeze=False)
        for row_i, p_idx in enumerate(snapshot_prompt_idxs):
            row_lookup = {r[0]: (r[1], r[2]) for r in snapshots[p_idx]}
            for c_idx, rd in enumerate(rounds_present):
                ax = axes[row_i][c_idx]
                ax.set_xticks([]); ax.set_yticks([])
                if rd in row_lookup:
                    ir, ip = row_lookup[rd]
                    try:
                        ax.imshow(Image.open(ip))
                        ax.set_title(f"r{rd}\nIR={ir:+.2f}", fontsize=7)
                    except Exception:
                        ax.text(0.5, 0.5, "missing", ha="center", va="center")
                if c_idx == 0:
                    ax.set_ylabel(all_prompts[p_idx][:25], fontsize=8, rotation=0,
                                  ha="right", va="center")
        grid_path = SAVE_DIR / "snapshot_grid.png"
        fig.suptitle("Per-prompt evolution over training (fixed seed)", y=1.0, fontsize=10)
        fig.tight_layout()
        fig.savefig(grid_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"samples/snapshot_grid": wandb.Image(str(grid_path))})
        print(f"Snapshot grid: {grid_path}")

wandb.finish()
print("Done.")
