import torch
from diffusers import StableDiffusionPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=pipe_dtype,
).to(device)
pipe.enable_attention_slicing()

prompt = (
#     "a casually dressed woman with shoulder-length dark hair, wearing thin round glasses, "
#     "a fitted black t-shirt, high-waisted light blue jeans, and white running shoes; gently holding a small brown terrier mix with floppy ears; "
#     "the dog resting calmly against her chest; standing upright in an outdoor park setting, soft natural daylight"
)


def sample_t2i(num_steps=100, guidance_scale=7.5, seed=10):
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


img = sample_t2i()
img.save("example_t2i.png")
