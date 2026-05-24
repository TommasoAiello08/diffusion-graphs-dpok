from graphviz import Digraph
from graphviz.backend import ExecutableNotFound
import textwrap


def render_reward_graph_weighted(
    nodes,
    bindings,
    interactions,
    dbg=None,
    weights=None,
    out_path="reward_graph_weighted",
    keep_dot=False,
):
    """
    Renders and saves a weighted reward graph as PNG.
    Returns the saved PNG path.

    Requires:
      - python package: graphviz
      - system binary: graphviz (dot)
    """

    # ---- weights ----
    w_node = float(weights.get("node", 1.0)) if weights else 1.0
    w_bind = float(weights.get("binding", 1.0)) if weights else 1.0
    w_edge = float(weights.get("interaction", 1.0)) if weights else 1.0

    # ---- pull debug tensors to python lists ----
    def _to_list(x):
        try:
            return x.detach().float().cpu().tolist()
        except Exception:
            return list(x)

    def _vec(size, *keys, default=0.0):
        if isinstance(dbg, dict):
            for key in keys:
                if key in dbg:
                    values = _to_list(dbg[key])
                    if len(values) == size:
                        return [float(v) for v in values]
        return [float(default)] * size

    phi_node = _vec(len(nodes), "phi_node", "b_node", default=0.0)
    phi_bind = _vec(len(bindings), "phi_bind", "b_bind", default=0.0)
    phi_edge = _vec(len(interactions), "phi_edge", "b_edge", default=0.0)
    g_node = _vec(len(nodes), "gate_node", "g_node", default=1.0)

    node_index = {n.name: i for i, n in enumerate(nodes)}

    # ---- helpers ----
    def wrap(s, width=28):
        return textwrap.fill(str(s), width)

    def clamp01(x):
        return max(0.0, min(1.0, float(x)))

    def blend(c0, c1, t):
        """Blend two hex colors c0->c1 with t in [0,1]."""
        t = clamp01(t)
        c0 = c0.lstrip("#")
        c1 = c1.lstrip("#")
        r0, g0, b0 = int(c0[0:2], 16), int(c0[2:4], 16), int(c0[4:6], 16)
        r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
        r = int(round(r0 + (r1 - r0) * t))
        g = int(round(g0 + (g1 - g0) * t))
        b = int(round(b0 + (b1 - b0) * t))
        return f"#{r:02x}{g:02x}{b:02x}"

    def diverging_color(x, scale):
        """Light gray -> green for +, light gray -> red for -."""
        scale = float(scale)
        if scale <= 1e-8:
            return "#dddddd"
        t = clamp01(abs(float(x)) / scale)
        return blend("#eeeeee", "#2ca02c", t) if x >= 0 else blend("#eeeeee", "#d62728", t)

    def as_float(x):
        """Tensor/number -> python float"""
        try:
            return float(x.detach().cpu().item())
        except Exception:
            return float(x)

    # ---- contributions ----
    node_contrib = [w_node * x for x in phi_node]

    bind_contrib = []
    for bi, b in enumerate(bindings):
        ni = node_index[b.node]
        bind_contrib.append(g_node[ni] * w_bind * phi_bind[bi])

    edge_contrib = []
    for ei, e in enumerate(interactions):
        si = node_index[e.src]
        di = node_index[e.dst]
        edge_contrib.append(g_node[si] * g_node[di] * w_edge * phi_edge[ei])

    scale_bind = max(1e-6, max((abs(x) for x in bind_contrib), default=1.0))
    scale_edge = max(1e-6, max((abs(x) for x in edge_contrib), default=1.0))

    # ---- build graph ----
    dot = Digraph("RewardGraph", format="png")
    dot.attr(rankdir="LR", splines="spline", bgcolor="white")
    dot.attr("node", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica")

    def _scalar(*keys, default=0.0):
        if isinstance(dbg, dict):
            for key in keys:
                if key in dbg:
                    return as_float(dbg[key])
        return float(default)

    # Summary box
    Rn = _scalar("R_node", default=sum(node_contrib))
    Rb = _scalar("R_bind", default=sum(bind_contrib))
    Re = _scalar("R_edge", default=sum(edge_contrib))
    Rt = Rn + Rb + Re

    summary_label = (
        f"R_total: {Rt:+.3f}\n"
        f"R_node: {Rn:+.3f}\n"
        f"R_bind: {Rb:+.3f}\n"
        f"R_edge: {Re:+.3f}"
    )
    dot.node("SUMMARY", label=summary_label, shape="box", style="rounded,filled", fillcolor="#f7f7f7")

    # Entities
    with dot.subgraph(name="cluster_entities") as c:
        c.attr(label="Entities", color="#DDDDDD", style="rounded")
        for n in nodes:
            i = node_index[n.name]
            g = float(g_node[i])
            fill = blend("#e6e6e6", "#1f77b4", g)

            label = (
                f"{n.name}\n"
                f"g={g:.2f}  phi={phi_node[i]:+.2f}\n"
                f"w*phi={node_contrib[i]:+.2f}\n"
                f"{wrap(n.exist_prompt)}"
            )

            c.node(
                n.name,
                label=label,
                shape="ellipse",
                style="filled",
                fillcolor=fill,
                fontcolor="white" if g > 0.55 else "black",
            )

    # Bindings
    bindings_by_node = {}
    for bi, b in enumerate(bindings):
        bindings_by_node.setdefault(b.node, []).append((bi, b))

    for parent, blist in bindings_by_node.items():
        with dot.subgraph(name=f"cluster_bind_{parent}") as c:
            c.attr(label=f"Bindings: {parent}", color="#EEEEEE", style="rounded,dashed")
            for bi, b in blist:
                ni = node_index[parent]
                contrib = float(bind_contrib[bi])
                fill = diverging_color(contrib, scale_bind)

                b_id = f"bind_{parent}_{bi}"
                label = (
                    f"B{bi}  contrib={contrib:+.2f}\n"
                    f"g={g_node[ni]:.2f}  phi={phi_bind[bi]:+.2f}\n"
                    f"{wrap(b.prompt)}"
                )

                c.node(b_id, label=label, shape="box", style="rounded,filled", fillcolor=fill)
                dot.edge(parent, b_id, arrowhead="none", color="#999999")

    # Interactions
    for ei, e in enumerate(interactions):
        contrib = float(edge_contrib[ei])
        color = diverging_color(contrib, scale_edge)
        penwidth = 1.0 + 3.0 * clamp01(abs(contrib) / scale_edge)

        label = (
            f"I{ei}  contrib={contrib:+.2f}\n"
            f"g={g_node[node_index[e.src]]:.2f}·{g_node[node_index[e.dst]]:.2f}  "
            f"phi={phi_edge[ei]:+.2f}\n"
            f"{wrap(e.prompt, 34)}"
        )

        dot.edge(e.src, e.dst, label=label, color=color, penwidth=str(penwidth), arrowsize="0.8")

    # Attach summary (guard empty)
    if nodes:
        dot.edge("SUMMARY", nodes[0].name, style="dashed", arrowhead="none", color="#bbbbbb")

    # ---- SAVE PNG ----
    try:
        output_path = dot.render(filename=out_path, format="png", cleanup=not keep_dot)
    except ExecutableNotFound as ex:
        raise RuntimeError(
            "Graphviz 'dot' executable not found. Install system graphviz.\n"
            "Ubuntu/Debian: sudo apt-get install graphviz\n"
            "Conda: conda install -c conda-forge graphviz"
        ) from ex

    print(f"Saved graph to: {output_path}")
    return output_path
