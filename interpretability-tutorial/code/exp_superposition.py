"""Session 4: superposition in a toy model that is trained from scratch (about two minutes on a CPU).

The model is the one of Elhage et al. (2022): x = W f, f_hat = ReLU(W^T x + b), with sparse features f.
Writes code/results_superposition.json and figures/fig_toy_geometry.pdf, fig_toy_capacity.pdf,
fig_random_cosines.pdf.
"""
import json
from pathlib import Path

import numpy as np
import torch

from plotstyle import plt, BLUE, RED, MUTED, INK, grid

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
torch.set_num_threads(2)


def sample(batch, n_features, density):
    """Each feature is zero with probability 1 - density and uniform on [0, 1] otherwise."""
    return torch.rand(batch, n_features) * (torch.rand(batch, n_features) < density)


def train(n_features, dim, density, importance, seed, steps=6000, batch=2048):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(dim, n_features) * 0.5)
    b = torch.nn.Parameter(torch.zeros(n_features))
    optimiser = torch.optim.Adam([W, b], lr=1e-2)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    loss_of = lambda f: (importance * (f - torch.relu(f @ W.T @ W + b)) ** 2).sum(-1).mean()
    for _ in range(steps):
        loss = loss_of(sample(batch, n_features, density))
        optimiser.zero_grad(); loss.backward(); optimiser.step(); schedule.step()
    with torch.no_grad():
        final = np.mean([loss_of(sample(100_000, n_features, density)).item() for _ in range(5)])
    return W.detach().numpy(), b.detach().numpy(), float(final)


def best(n_features, dim, density, importance, seeds=4, **kwargs):
    return min((train(n_features, dim, density, importance, seed, **kwargs) for seed in range(seeds)), key=lambda r: r[2])


results = {}

# ---- A. Two features, one dimension: compare with the two hand-made solutions of the text -----------
densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
two = []
for density in densities:
    W, b, loss = best(2, 1, density, torch.ones(2))
    two.append({"density": density, "trained_loss": loss, "W": W.round(3).tolist(), "b": b.round(3).tolist(),
                "dedicated": density / 3 - density ** 2 / 4, "antipodal": density ** 2 / 3})
results["two_features_one_dimension"] = two

# ---- B. Five features, two dimensions, as in the first figure of Elhage et al. -----------------------
importance5 = 0.7 ** torch.arange(5)
geometry = {}
for density in (1.0, 0.2, 0.05):
    W, b, loss = best(5, 2, density, importance5, steps=10000)
    geometry[str(density)] = {"W": W.round(3).tolist(), "b": b.round(3).tolist(), "loss": loss,
                              "norms": np.linalg.norm(W, axis=0).round(3).tolist()}
results["five_features_two_dimensions"] = geometry

# ---- C. Twenty features, five dimensions: how many features get a direction? ------------------------
importance20 = 0.9 ** torch.arange(20)
capacity = []
for density in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01):
    W, b, loss = best(20, 5, density, importance20, seeds=3, steps=10000)
    norms = np.linalg.norm(W, axis=0)
    capacity.append({"density": density, "represented": int((norms > 0.5).sum()), "norms": norms.round(2).tolist()})
results["twenty_features_five_dimensions"] = capacity

# ---- D. Random directions in 768 dimensions are almost orthogonal -----------------------------------
torch.manual_seed(0)
vectors = torch.nn.functional.normalize(torch.randn(2000, 768), dim=-1)
cosines = (vectors @ vectors.T)[np.triu_indices(2000, k=1)].numpy()
results["random_directions"] = {"n_vectors": 2000, "dimension": 768, "std": float(cosines.std()),
                                "one_over_sqrt_d": 768 ** -0.5, "max_abs": float(np.abs(cosines).max()),
                                "share_above_0.1": float((np.abs(cosines) > 0.1).mean())}
(ROOT / "code" / "results_superposition.json").write_text(json.dumps(results, indent=1))

# ---------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
for ax, (density, entry) in zip(axes, geometry.items()):
    W = np.array(entry["W"])
    ax.axhline(0, color="#d8d7d2", lw=0.6); ax.axvline(0, color="#d8d7d2", lw=0.6)
    for j in range(5):
        ax.annotate("", xy=W[:, j], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=9))
        if np.linalg.norm(W[:, j]) > 0.15:
            ax.text(*(W[:, j] * 1.18), str(j + 1), ha="center", va="center", fontsize=8, color=INK)
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    share = f"{100 * float(density):.0f}"
    ax.set_title(f"features active {share}% of the time", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig_toy_geometry.pdf"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
ax = axes[0]
d = np.linspace(0.05, 1, 100)
ax.plot(d, d / 3 - d ** 2 / 4, color=MUTED, lw=1.2, ls="--", label="dedicated (formula)")
ax.plot(d, d ** 2 / 3, color=RED, lw=1.2, label="antipodal (formula)")
ax.plot(densities, [t["trained_loss"] for t in two], color=BLUE, lw=0, marker="o", ms=4, label="trained model")
ax.set_xlabel("probability $\\delta$ that a feature is active"); ax.set_ylabel("loss"); ax.legend(fontsize=8); grid(ax)
ax = axes[1]
ax.semilogx([c["density"] for c in capacity], [c["represented"] for c in capacity], color=BLUE, lw=1.6, marker="o", ms=4)
ax.axhline(5, color=MUTED, lw=0.8, ls="--"); ax.text(0.011, 5.4, "number of dimensions", fontsize=8, color=MUTED)
ax.set_xlabel("probability $\\delta$ that a feature is active"); ax.set_ylabel("features with a direction (of 20)")
ax.set_ylim(0, 21); grid(ax)
fig.tight_layout(w_pad=2.0); fig.savefig(FIG / "fig_toy_capacity.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 2.3))
ax.hist(cosines, bins=120, color=BLUE)
for sign in (-1, 1):
    ax.axvline(sign * 768 ** -0.5, color=RED, lw=1.0, ls="--")
ax.set_xlabel("inner product of two random unit vectors, $d=768$"); ax.set_ylabel("number of pairs"); ax.set_xlim(-0.2, 0.2)
fig.tight_layout(); fig.savefig(FIG / "fig_random_cosines.pdf"); plt.close(fig)

for t in two:
    print(t)
print({k: (v["norms"], round(v["loss"], 4)) for k, v in geometry.items()})
print([(c["density"], c["represented"]) for c in capacity])
print(results["random_directions"])
