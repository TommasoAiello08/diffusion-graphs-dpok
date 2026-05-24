from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from refgraphs import EXACT
    from rewards.clip import CLIPScorer as ProjectCLIPScorer
    from rewards.rewards import RewardGraph as ProjectRewardGraph
except ImportError:
    EXACT = None
    ProjectCLIPScorer = None
    ProjectRewardGraph = None

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
    "Project-CLIPScorer": None,
    "Project-RewardModel": None,
    "Project-RewardModelConfig": None,
}


def _import_openai_clip():
    try:
        import clip as openai_clip
    except ImportError as exc:
        raise ImportError(
            "The `clip` package is required for `Clip-Score`. "
            "Install it in your venv, e.g. `pip install git+https://github.com/openai/CLIP.git`."
        ) from exc
    return openai_clip


def _import_rm_load():
    try:
        from .image_reward_utils import rm_load as _rm_load
    except ImportError:
        if __package__:
            raise
        from image_reward_utils import rm_load as _rm_load
    return _rm_load


def _float_score(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


class OpenCLIPScoreFallback(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        if ProjectCLIPScorer is None:
            raise ImportError(
                "OpenAI CLIP is unavailable and fallback scorer `rewards.clip.CLIPScorer` "
                "could not be imported."
            )
        self.device = _resolve_device(device)
        self.scorer = ProjectCLIPScorer().to(self.device)

    def score(self, prompt, pil_image, return_feature=False):
        image = _to_chw_tensor(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.scorer.encode_image(image)
            reward = self.scorer.score_from_image_feature(image_features, prompt)

        if return_feature:
            return reward, {"image": image_features, "txt": None}

        return _float_score(reward)


def _get_clip_score_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if REWARDS_DICT["Clip-Score"] is not None:
        return REWARDS_DICT["Clip-Score"]

    def _build_openai_clip(target_device):
        try:
            return CLIPScore(download_root=".", device=target_device)
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc) and target_device == "cuda":
                return CLIPScore(download_root=".", device="cpu")
            raise

    def _build_open_clip_fallback(target_device):
        try:
            return OpenCLIPScoreFallback(device=target_device)
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc) and target_device == "cuda":
                return OpenCLIPScoreFallback(device="cpu")
            raise

    try:
        model = _build_openai_clip(device)
    except ImportError:
        model = _build_open_clip_fallback(device)
    REWARDS_DICT["Clip-Score"] = model
    return model


# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(
    reward_name,
    images,
    prompts,
    metric_to_chase="overall_score",
    reward_kwargs: Optional[Dict[str, Any]] = None,
):
    reward_kwargs = reward_kwargs or {}
    prompts = _normalize_prompts(prompts, len(images))
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts)

    elif reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts)

    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    elif reward_name == "LLMGrader":
        return do_llm_grading(
            images=images, prompts=prompts, metric_to_chase=metric_to_chase
        )

    elif reward_name in ("Project-RewardGraph", "RewardGraph"):
        return do_project_reward_graph(
            images=images, prompts=prompts, **reward_kwargs
        )

    else:
        raise ValueError(f"Unknown metric: {reward_name}")


def _normalize_prompts(prompts, num_images):
    if isinstance(prompts, str):
        return [prompts] * num_images
    if prompts is None:
        return [""] * num_images
    prompts = list(prompts)
    if len(prompts) == num_images:
        return prompts
    if len(prompts) == 1 and num_images > 1:
        return prompts * num_images
    raise ValueError(
        f"Prompt count ({len(prompts)}) must be 1 or match image count ({num_images})."
    )


def _resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _to_chw_tensor(image):
    if isinstance(image, torch.Tensor):
        tensor = image.detach()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "Expected a single image tensor in 4D input, got batch size "
                    f"{tensor.shape[0]}."
                )
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(
                f"Unsupported tensor image rank {tensor.ndim}; expected 3D CHW or HWC."
            )

        if tensor.shape[0] in (1, 3):
            chw = tensor
        elif tensor.shape[-1] in (1, 3):
            chw = tensor.permute(2, 0, 1)
        else:
            raise ValueError(
                "Unsupported tensor image shape; expected CHW or HWC with 1 or 3 channels."
            )

        if chw.dtype == torch.uint8:
            chw = chw.float() / 255.0
        else:
            chw = chw.float()
        return chw.clamp(0.0, 1.0)

    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    if isinstance(image, np.ndarray):
        array = torch.from_numpy(image)
        if array.ndim == 2:
            array = array.unsqueeze(-1)
        if array.ndim != 3:
            raise ValueError(
                f"Unsupported ndarray image rank {array.ndim}; expected HWC or CHW."
            )
        if array.shape[0] in (1, 3):
            chw = array
        elif array.shape[-1] in (1, 3):
            chw = array.permute(2, 0, 1)
        else:
            raise ValueError(
                "Unsupported ndarray image shape; expected CHW or HWC with 1 or 3 channels."
            )
        if chw.dtype == torch.uint8:
            chw = chw.float() / 255.0
        else:
            chw = chw.float()
            if chw.max() > 1.0:
                chw = chw / 255.0
        return chw.clamp(0.0, 1.0)

    raise TypeError(f"Unsupported image type: {type(image)}")


def _images_to_batch_tensor(images: Iterable, device: torch.device):
    image_tensors = [_to_chw_tensor(image) for image in images]
    return torch.stack(image_tensors, dim=0).to(device)


def _get_project_reward_model(
    *,
    device: torch.device,
    graph,
    weights,
    gating,
    alpha,
    gate_tau,
    contrastive_temp,
    contrastive_mode,
    hard_threshold,
    mp_iters,
    mp_eta,
):
    global REWARDS_DICT

    if ProjectCLIPScorer is None or ProjectRewardGraph is None:
        raise ImportError(
            "Project reward dependencies are unavailable. Make sure this repository's "
            "`src` directory is on PYTHONPATH."
        )

    scorer = REWARDS_DICT["Project-CLIPScorer"]
    if scorer is None:
        scorer = ProjectCLIPScorer().to(device)
        REWARDS_DICT["Project-CLIPScorer"] = scorer
    else:
        scorer = scorer.to(device)

    config = (
        id(graph),
        tuple(sorted(weights.items())),
        str(device),
        gating,
        float(alpha),
        float(gate_tau),
        float(contrastive_temp),
        contrastive_mode,
        float(hard_threshold),
        int(mp_iters),
        float(mp_eta),
    )
    if REWARDS_DICT["Project-RewardModelConfig"] != config:
        REWARDS_DICT["Project-RewardModel"] = ProjectRewardGraph(
            scorer=scorer,
            graph=graph,
            weights=weights,
            gating=gating,
            alpha=alpha,
            gate_tau=gate_tau,
            contrastive_temp=contrastive_temp,
            contrastive_mode=contrastive_mode,
            hard_threshold=hard_threshold,
            mp_iters=mp_iters,
            mp_eta=mp_eta,
        ).to(device)
        REWARDS_DICT["Project-RewardModelConfig"] = config

    return REWARDS_DICT["Project-RewardModel"]


def do_project_reward_graph(
    *,
    images,
    prompts,
    graph=None,
    weights=None,
    device=None,
    gating="soft",
    alpha=10.0,
    gate_tau=0.5,
    contrastive_temp=0.03,
    contrastive_mode="prob",
    hard_threshold=0.0,
    mp_iters=2,
    mp_eta=0.7,
):
    _ = prompts  # Kept for API consistency with other reward functions.

    if graph is None:
        if EXACT is None:
            raise ImportError(
                "Default reference graph is unavailable. Provide `graph` explicitly."
            )
        graph = EXACT

    if weights is None:
        weights = {"node": 1.0, "binding": 1.0, "interaction": 6.0}

    reward_device = _resolve_device(device)
    reward_model = _get_project_reward_model(
        device=reward_device,
        graph=graph,
        weights=weights,
        gating=gating,
        alpha=alpha,
        gate_tau=gate_tau,
        contrastive_temp=contrastive_temp,
        contrastive_mode=contrastive_mode,
        hard_threshold=hard_threshold,
        mp_iters=mp_iters,
        mp_eta=mp_eta,
    )
    image_batch = _images_to_batch_tensor(images, reward_device)

    with torch.no_grad():
        rewards = []
        for idx in range(image_batch.shape[0]):
            reward_value, _ = reward_model(image_batch[idx : idx + 1])
            rewards.append(float(reward_value.detach().cpu()))

    return rewards


# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    if use_paths:
        scores = hpsv2.score(images, prompts, hps_version="v2.1")
        scores = [float(score) for score in scores]
    else:
        scores = []
        for i, image in enumerate(images):
            score = hpsv2.score(image, prompts[i], hps_version="v2.1")
            # print(f"Human preference score for image {i}: {score}")
            score = float(score[0])
            scores.append(score)

    # print(f"Human preference scores: {scores}")
    return scores

# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    clip_model = _get_clip_score_model()
    with torch.no_grad():
        arr_clip_result = []
        arr_img_features = []
        for i, prompt in enumerate(prompts):
            clip_result, feature_vect = clip_model.score(
                prompt, images[i], return_feature=True
            )

            arr_clip_result.append(_float_score(clip_result))
            arr_img_features.append(feature_vect['image'])

    # calculate diversity by computing pairwise similarity between image features
    feature_device = arr_img_features[0].device if arr_img_features else torch.device("cpu")
    diversity = torch.zeros(len(images), len(images), device=feature_device)
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diversity[i, j] = (arr_img_features[i] - arr_img_features[j]).pow(2).sum()
            diversity[j, i] = diversity[i, j]
    n_samples = len(images)
    diversity = diversity.sum() / (n_samples * (n_samples - 1))

    return arr_clip_result, diversity.item()

# Compute ImageReward
def do_image_reward(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        rm_load = _import_rm_load()
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    with torch.no_grad():
        image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
        # image_reward_result = [REWARDS_DICT["ImageReward"].score(prompt, images[i]) for i, prompt in enumerate(prompts)]

    return image_reward_result

# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    clip_model = _get_clip_score_model()
    with torch.no_grad():
        clip_result = [
            clip_model.score(prompt, images[i])
            for i, prompt in enumerate(prompts)
        ]
    return clip_result


# Compute LLM-grading
def do_llm_grading(*, images, prompts, metric_to_chase="overall_score"):
    global REWARDS_DICT
    
    if REWARDS_DICT["LLMGrader"] is None:
        REWARDS_DICT["LLMGrader"]  = LLMGrader()
    llm_grading_result = [
        REWARDS_DICT["LLMGrader"].score(images=images[i], prompts=prompt, metric_to_chase=metric_to_chase)
        for i, prompt in enumerate(prompts)
    ]
    return llm_grading_result


'''
@File       :   CLIPScore.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore.
* Based on CLIP code base
* https://github.com/openai/CLIP
'''


class CLIPScore(nn.Module):
    def __init__(self, download_root, device='cpu'):
        super().__init__()
        clip = _import_openai_clip()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )
        self._clip = clip

        if device == "cpu":
            self.clip_model.float()
        else:
            self._clip.model.convert_weights(
                self.clip_model
            )  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # if (type(image_path).__name__=='list'):
        #     _, rewards = self.inference_rank(prompt, image_path)
        #     return rewards

        # text encode
        text = self._clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))

        # score
        rewards = torch.sum(
            torch.mul(txt_features, image_features), dim=1, keepdim=True
        )

        if return_feature:
            return rewards, {'image': image_features, 'txt': txt_features}

        return rewards.detach().cpu().numpy().item()
