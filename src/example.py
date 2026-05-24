import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from refgraphs import EXACT
from rewards.clip import CLIPScorer
from rewards.rewards import SimpleHierReward, RewardGraph

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=pipe_dtype,
).to(device)
pipe.enable_attention_slicing()

weights = {"node": 1.0, "binding": 1.0, "interaction": 6.0}
scorer = CLIPScorer().to(device)
reward = RewardGraph(
    scorer, EXACT,
    weights=weights,
    gating="soft", alpha=10.0, gate_tau=0.5, contrastive_temp=0.03, contrastive_mode="prob",
).to(device)
prompt = "A casually dressed man wearing rectangular glasses, a white t-shirt, blue jeans, bright red sneakers, gently holding a fluffy cat in his arms"


def sample_with_reward(num_steps=100, alpha=0.1, save_steps=True, steps_dir="example_steps"):
    sample_size = pipe.unet.config.sample_size
    if isinstance(sample_size, int):
        latent_h = sample_size
        latent_w = sample_size
    else:
        latent_h, latent_w = sample_size

    step_dir = None
    if save_steps:
        step_dir = Path(steps_dir)
        step_dir.mkdir(parents=True, exist_ok=True)

    latents = torch.randn(
        (1, pipe.unet.config.in_channels, latent_h, latent_w),
        device=device,
        dtype=pipe.unet.dtype,
    )

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=device)

    for step_idx, t in enumerate(scheduler.timesteps):
        latents = latents.detach().requires_grad_(True)
        with torch.no_grad():
            latent_model_input = scheduler.scale_model_input(latents, t)
            noise_pred = pipe.unet(
                latent_model_input, t, encoder_hidden_states=prompt_embeds
            ).sample

        # Tweedie-style x0 estimate from (x_t, eps_theta) before decoding for reward.
        t_index = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        alpha_prod_t = scheduler.alphas_cumprod[t_index].to(
            device=latents.device, dtype=latents.dtype
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
        if pred_type == "epsilon":
            pred_x0 = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        elif pred_type == "v_prediction":
            pred_x0 = alpha_prod_t.sqrt() * latents - beta_prod_t.sqrt() * noise_pred
        elif pred_type == "sample":
            pred_x0 = noise_pred
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        approx_img = pipe.vae.decode(
            pred_x0 / pipe.vae.config.scaling_factor, return_dict=False
        )[0]
        approx_img_for_reward = ((approx_img / 2) + 0.5).clamp(0, 1).to(torch.float32)

        R, _ = reward(approx_img_for_reward)
        grad = torch.autograd.grad(
            R,
            latents,
            create_graph=False,
            retain_graph=False,
        )[0].detach()

        noise_pred = noise_pred - alpha * grad.to(noise_pred.dtype)

        with torch.no_grad():
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            if step_dir is not None:
                step_image = pipe.image_processor.postprocess(
                    approx_img.detach(), output_type="pil"
                )[0]
                step_image.save(step_dir / f"step_{step_idx:03d}.png")

    with torch.no_grad():
        image = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False
        )[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]

# Example
img = sample_with_reward()
img.save("example.png")
