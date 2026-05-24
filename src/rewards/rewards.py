from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
from rewards.clip import CLIPScorer

@dataclass
class Node:
    name: str
    exist_prompt: str  # e.g. "a man"
    neg_prompts: List[str] = field(default_factory=list)

@dataclass
class Binding:
    node: str          # node name
    prompt: str        # e.g. "a blue t-shirt"
    neg_prompts: List[str] = field(default_factory=list)

@dataclass
class Interaction:
    src: str
    dst: str
    prompt: str        # e.g. "a man holding a cat"
    neg_prompts: List[str] = field(default_factory=list)

@dataclass
class Graph: 
    nodes: List[Node]
    bindings: List[Binding]
    interactions: List[Interaction]

class SimpleHierReward(torch.nn.Module):
    def __init__(
        self,
        scorer: CLIPScorer,
        graph: Graph,
        weights: Optional[Dict[str, float]] = None,
        gating: str = "soft",          # "soft" or "hard"
        alpha: float = 10.0,           # soft gate sharpness
        gate_tau: float = 0.5,         # soft gate center
        contrastive_temp: float = 0.05, # pos-vs-neg sharpness
        contrastive_mode: str = "prob", # "prob" (positive) or "margin"
        hard_threshold: float = 0.0,   # hard gate threshold
    ):
        super().__init__()
        assert gating in ("soft", "hard")
        assert contrastive_mode in ("prob", "margin")
        self.scorer = scorer
        self.graph = graph

        self.gating = gating
        self.alpha = alpha
        self.gate_tau = gate_tau
        self.contrastive_temp = contrastive_temp
        self.contrastive_mode = contrastive_mode
        self.hard_threshold = hard_threshold

        self.w_node = float(weights.get("node", 1.0)) if weights else 1.0
        self.w_bind = float(weights.get("binding", 1.0)) if weights else 1.0
        self.w_edge = float(weights.get("interaction", 1.0)) if weights else 1.0

        self.node_index = {n.name: i for i, n in enumerate(self.graph.nodes)}

    def _gate(self, phi_node: torch.Tensor) -> torch.Tensor:
        if self.gating == "soft":
            return torch.sigmoid(self.alpha * (phi_node - self.gate_tau))
        # hard gate: detach to avoid weird gradients through thresholding
        return (phi_node.detach() > self.hard_threshold).float()

    def _contrastive_potential(
        self,
        image_feat: torch.Tensor,
        pos_prompt: str,
        neg_prompts: List[str],
    ) -> torch.Tensor:
        temp = max(float(self.contrastive_temp), 1e-6)
        pos_score = self.scorer.score_from_image_feature(image_feat, pos_prompt)
        clean_negs = [p for p in dict.fromkeys(neg_prompts) if p != pos_prompt]
        if not clean_negs:
            if self.contrastive_mode == "prob":
                return torch.ones_like(pos_score)
            return pos_score

        neg_scores = self.scorer.score_texts_from_image_feature(image_feat, clean_negs)
        if self.contrastive_mode == "prob":
            logits = torch.cat([pos_score.unsqueeze(0), neg_scores], dim=0) / temp
            return torch.softmax(logits, dim=0)[0]

        return pos_score - temp * torch.logsumexp(neg_scores / temp, dim=0)

    def forward(self, image: torch.Tensor):
        device = image.device
        image_feat = self.scorer.encode_image(image)

        # Potentials (one scorer, many texts)
        phi_node = torch.stack(
            [
                self._contrastive_potential(image_feat, n.exist_prompt, n.neg_prompts)
                for n in self.graph.nodes
            ]
        )  # (N,)
        phi_bind = torch.stack(
            [
                self._contrastive_potential(image_feat, b.prompt, b.neg_prompts)
                for b in self.graph.bindings
            ]
        )  # (B,)
        phi_edge = torch.stack(
            [
                self._contrastive_potential(image_feat, e.prompt, e.neg_prompts)
                for e in self.graph.interactions
            ]
        )  # (E,)

        g = self._gate(phi_node)  # (N,)

        # Node reward
        R_node = self.w_node * phi_node.sum()

        # Binding reward (gated by its node)
        R_bind = torch.zeros((), device=device)
        for bi, b in enumerate(self.graph.bindings):
            ni = self.node_index[b.node]
            R_bind = R_bind + self.w_bind * g[ni] * phi_bind[bi]

        # Interaction reward (gated by both endpoints)
        R_edge = torch.zeros((), device=device)
        for ei, e in enumerate(self.graph.interactions):
            si = self.node_index[e.src]
            di = self.node_index[e.dst]
            R_edge = R_edge + self.w_edge * (g[si] * g[di]) * phi_edge[ei]

        R = R_node + R_bind + R_edge

        debug = {
            "phi_node": phi_node, "phi_bind": phi_bind, "phi_edge": phi_edge,
            "gate_node": g,
            "gate_tau": torch.tensor(self.gate_tau, device=device),
            "contrastive_temp": torch.tensor(self.contrastive_temp, device=device),
            "contrastive_mode_prob": torch.tensor(
                1.0 if self.contrastive_mode == "prob" else 0.0, device=device
            ),
            "R_node": R_node.detach(), "R_bind": R_bind.detach(), "R_edge": R_edge.detach(),
        }
        return R, debug



class RewardGraph(torch.nn.Module):
    def __init__(
        self,
        scorer: CLIPScorer,
        graph: Graph,
        weights: Optional[Dict[str, float]] = None,
        gating: str = "soft",          # "soft" or "hard"
        alpha: float = 10.0,           # soft gate sharpness
        gate_tau: float = 0.5,         # soft gate center
        contrastive_temp: float = 0.05, # pos-vs-neg sharpness
        contrastive_mode: str = "prob", # "prob" (positive) or "margin"
        hard_threshold: float = 0.0,   # hard gate threshold
        mp_iters: int = 2,             # message passing iterations
        mp_eta: float = 0.7            # update step-size
    ):
        super().__init__()
        assert gating in ("soft", "hard")
        assert contrastive_mode in ("prob", "margin")
        self.scorer = scorer
        self.graph = graph

        self.gating = gating
        self.alpha = alpha
        self.gate_tau = gate_tau
        self.contrastive_temp = contrastive_temp
        self.contrastive_mode = contrastive_mode
        self.hard_threshold = hard_threshold
        self.mp_iters = mp_iters
        self.mp_eta = mp_eta

        # Default weights
        self.w_node = 1.0
        self.w_bind = 1.0
        self.w_edge = 1.0
        if weights:
            self.w_node = float(weights.get("node", self.w_node))
            self.w_bind = float(weights.get("binding", self.w_bind))
            self.w_edge = float(weights.get("interaction", self.w_edge))

        self.node_index = {n.name: i for i, n in enumerate(self.graph.nodes)}

        # adjacency for interactions
        self.inc_edges: Dict[str, List[int]] = {n.name: [] for n in self.graph.nodes}
        for ei, e in enumerate(self.graph.interactions):
            self.inc_edges[e.src].append(ei)
            self.inc_edges[e.dst].append(ei)

        # bindings per node
        self.inc_bind: Dict[str, List[int]] = {n.name: [] for n in self.graph.nodes}
        for bi, b in enumerate(self.graph.bindings):
            self.inc_bind[b.node].append(bi)

    def _gate(self, belief: torch.Tensor) -> torch.Tensor:
        if self.gating == "soft":
            return torch.sigmoid(self.alpha * (belief - self.gate_tau))
        # hard gate: non-differentiable step (grad = 0 through the gate)
        return (belief.detach() > self.hard_threshold).float()

    def _contrastive_potential(
        self,
        image_feat: torch.Tensor,
        pos_prompt: str,
        neg_prompts: List[str],
    ) -> torch.Tensor:
        temp = max(float(self.contrastive_temp), 1e-6)
        pos_score = self.scorer.score_from_image_feature(image_feat, pos_prompt)
        clean_negs = [p for p in dict.fromkeys(neg_prompts) if p != pos_prompt]
        if not clean_negs:
            if self.contrastive_mode == "prob":
                return torch.ones_like(pos_score)
            return pos_score

        neg_scores = self.scorer.score_texts_from_image_feature(image_feat, clean_negs)
        if self.contrastive_mode == "prob":
            logits = torch.cat([pos_score.unsqueeze(0), neg_scores], dim=0) / temp
            return torch.softmax(logits, dim=0)[0]

        return pos_score - temp * torch.logsumexp(neg_scores / temp, dim=0)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device = image.device
        N = len(self.graph.nodes)
        B = len(self.graph.bindings)
        E = len(self.graph.interactions)
        image_feat = self.scorer.encode_image(image)

        # --- Potentials from the single scorer (CLIP or any VLM similarity) ---
        phi_node = torch.stack(
            [
                self._contrastive_potential(image_feat, n.exist_prompt, n.neg_prompts)
                for n in self.graph.nodes
            ]
        )  # (N,)
        phi_bind = torch.stack(
            [
                self._contrastive_potential(image_feat, b.prompt, b.neg_prompts)
                for b in self.graph.bindings
            ]
        )  # (B,)
        phi_edge = torch.stack(
            [
                self._contrastive_potential(image_feat, e.prompt, e.neg_prompts)
                for e in self.graph.interactions
            ]
        )  # (E,)

        # --- Initialize beliefs (start from local evidence) ---
        b_node = phi_node.clone()
        b_bind = phi_bind.clone()
        b_edge = phi_edge.clone()

        # --- Message passing iterations ---
        # Messages are simple additive influences (not exact BP, but behaves like it).
        for _ in range(self.mp_iters):
            # Node gates
            g_node = self._gate(b_node)  # (N,)

            # Binding -> node, node -> binding
            msg_bind_to_node = torch.zeros((N,), device=device)
            for bi, b in enumerate(self.graph.bindings):
                ni = self.node_index[b.node]
                # child (binding) contributes to parent (node) only if node is "on"
                msg_bind_to_node[ni] = msg_bind_to_node[ni] + g_node[ni] * b_bind[bi]

            msg_node_to_bind = torch.zeros((B,), device=device)
            for bi, b in enumerate(self.graph.bindings):
                ni = self.node_index[b.node]
                # parent supports child
                msg_node_to_bind[bi] = g_node[ni] * b_node[ni]

            # Edge -> nodes, nodes -> edge
            msg_edge_to_node = torch.zeros((N,), device=device)
            msg_node_to_edge = torch.zeros((E,), device=device)
            for ei, e in enumerate(self.graph.interactions):
                si = self.node_index[e.src]
                di = self.node_index[e.dst]
                # edge supports endpoints if both endpoints are on
                edge_gate = g_node[si] * g_node[di]
                msg_edge_to_node[si] = msg_edge_to_node[si] + edge_gate * b_edge[ei]
                msg_edge_to_node[di] = msg_edge_to_node[di] + edge_gate * b_edge[ei]
                # endpoints support edge
                msg_node_to_edge[ei] = edge_gate * (b_node[si] + b_node[di]) * 0.5

            # Update beliefs with damping (mp_eta)
            new_b_node = phi_node + 0.5 * msg_bind_to_node + 0.5 * msg_edge_to_node
            new_b_bind = phi_bind + 0.5 * msg_node_to_bind
            new_b_edge = phi_edge + 0.5 * msg_node_to_edge

            b_node = (1 - self.mp_eta) * b_node + self.mp_eta * new_b_node
            b_bind = (1 - self.mp_eta) * b_bind + self.mp_eta * new_b_bind
            b_edge = (1 - self.mp_eta) * b_edge + self.mp_eta * new_b_edge

        # --- Final hierarchical reward (gated sum) ---
        g_node = self._gate(b_node)

        R_node = self.w_node * b_node.sum()

        # Binding reward is gated by its node
        R_bind = torch.zeros((), device=device)
        for bi, b in enumerate(self.graph.bindings):
            ni = self.node_index[b.node]
            R_bind = R_bind + self.w_bind * g_node[ni] * b_bind[bi]

        # Edge reward gated by both endpoints
        R_edge = torch.zeros((), device=device)
        for ei, e in enumerate(self.graph.interactions):
            si = self.node_index[e.src]
            di = self.node_index[e.dst]
            R_edge = R_edge + self.w_edge * (g_node[si] * g_node[di]) * b_edge[ei]

        R = R_node + R_bind + R_edge

        debug = {
            "phi_node": phi_node, "phi_bind": phi_bind, "phi_edge": phi_edge,
            "b_node": b_node, "b_bind": b_bind, "b_edge": b_edge,
            "g_node": g_node,
            "gate_tau": torch.tensor(self.gate_tau, device=device),
            "contrastive_temp": torch.tensor(self.contrastive_temp, device=device),
            "contrastive_mode_prob": torch.tensor(
                1.0 if self.contrastive_mode == "prob" else 0.0, device=device
            ),
            "R_node": R_node.detach(), "R_bind": R_bind.detach(), "R_edge": R_edge.detach()
        }
        return R, debug
