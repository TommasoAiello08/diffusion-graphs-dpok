import open_clip
import torch
import torch.nn.functional as F


class CLIPScorer(torch.nn.Module):
    def __init__(self, model_name="ViT-L-14", pretrained="openai"):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode_text(self, text):
        tokens = self.tokenizer([text])
        return self.model.encode_text(tokens)

    def _interpolate_positional_embedding(
        self, pos_embed: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        cls_pos = pos_embed[:1]
        spatial_pos = pos_embed[1:]
        width = spatial_pos.shape[-1]

        old_grid = int(spatial_pos.shape[0] ** 0.5)
        if old_grid * old_grid != spatial_pos.shape[0]:
            raise ValueError("Expected a square CLIP positional embedding grid.")

        spatial_pos = spatial_pos.reshape(1, old_grid, old_grid, width).permute(0, 3, 1, 2)
        spatial_pos = F.interpolate(
            spatial_pos,
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
        )
        spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, width)
        return torch.cat([cls_pos, spatial_pos], dim=0)

    def _encode_image_variable_resolution(self, image_tensor: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        if not hasattr(visual, "positional_embedding"):
            return self.model.encode_image(image_tensor)

        x = visual.conv1(image_tensor)
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls = visual.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)

        pos_embed = visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        if x.shape[1] != pos_embed.shape[0]:
            pos_embed = self._interpolate_positional_embedding(pos_embed, grid_h, grid_w)

        x = x + pos_embed
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = visual.transformer(x)
        pooled, _ = visual._pool(x)

        if visual.proj is not None:
            pooled = pooled @ visual.proj
        return pooled

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_feat = self._encode_image_variable_resolution(image_tensor)
        return image_feat / image_feat.norm(dim=-1, keepdim=True)

    def score_texts_from_image_feature(
        self, image_feat: torch.Tensor, texts: list[str]
    ) -> torch.Tensor:
        if not texts:
            return torch.empty(0, device=image_feat.device, dtype=image_feat.dtype)

        text_tokens = self.tokenizer(texts).to(image_feat.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scores = image_feat @ text_feat.t()
        if scores.dim() == 2 and scores.shape[0] == 1:
            return scores[0]
        return scores

    def score_from_image_feature(self, image_feat: torch.Tensor, text: str) -> torch.Tensor:
        return self.score_texts_from_image_feature(image_feat, [text])[0]

    def score(self, image_tensor: torch.Tensor, text: str) -> torch.Tensor:
        image_feat = self.encode_image(image_tensor)
        return self.score_from_image_feature(image_feat, text)

