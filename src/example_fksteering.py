import torch
from diffusers import DDIMScheduler

from fk_steering.fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
from viz.viz_graph import render_reward_graph_weighted
from refgraphs import build_graph as build_graph
# from refgraphsteer import build_graph

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_dtype = torch.float16 if device == "cuda" else torch.float32

pipe = FKDStableDiffusion.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=pipe_dtype,
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

graph, prompt = build_graph(return_overall_prompt=True)
render_reward_graph_weighted(
    graph.nodes,
    graph.bindings,
    graph.interactions,
    dbg=None,
    out_path="example_fksteering_graph",
)
print("Prompt:", prompt)


def sample_fksteering(
    num_steps=100,
    guidance_scale=7.5,
    seed=10,
    num_particles=16,
    reward_name="Clip-Score",
):
    weights = {"node": 6.0, "binding": 6.0, "interaction": 6.0}
    fkd_args = {
        "use_smc": True,
        "guidance_reward_fn": reward_name,
        "potential_type": "diff",
        "lmbda": 8.0,
        "num_particles": num_particles,
        "adaptive_resampling": True,
        "resample_frequency": 3,
        "resampling_t_start": 0,
        "resampling_t_end": num_steps - 1,
        "time_steps": num_steps,
        "reward_min_value": 0.0,
        "device": torch.device(device),
    }
    if reward_name in ("Project-RewardGraph", "RewardGraph"):
        fkd_args["guidance_reward_kwargs"] = {
            "graph": graph,
            "weights": weights,
            "gating": "soft",
            "alpha": 10.0,
            "gate_tau": 0.5,
            "contrastive_temp": 0.03,
            "contrastive_mode": "prob",
            "mp_iters": 2,
            "mp_eta": 0.7,
        }

    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_particles,
        generator=generator,
        fkd_args=fkd_args,
    )
    return result.images[0]


sample_fksteering(reward_name="Clip-Score").save("example_fksteering.png")
sample_fksteering(reward_name="Project-RewardGraph").save("example_fksteering_reward_graph.png")
