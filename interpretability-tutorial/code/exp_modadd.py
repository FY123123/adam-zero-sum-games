"""Session 8: train a one-layer transformer on addition modulo 113 and look inside it.

Follows the setup of Nanda et al. (ICLR 2023): d = 128, 4 heads, 512 ReLU neurons, no LayerNorm,
30% of all pairs as training set, full-batch AdamW with weight decay 1. About 15 minutes on 4 CPU cores.
With the flag --replot the script reuses the saved run and only repeats the analysis and the figures.
Writes code/results_modadd.json and figures/fig_grokking.pdf, fig_fourier.pdf, fig_modadd_logits.pdf.
Usage:  python code/exp_modadd.py [epochs] [--replot]
"""
import json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from plotstyle import plt, BLUE, RED, MUTED, INK, grid

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
torch.manual_seed(0); torch.set_num_threads(4)
P, D, HEADS, D_HEAD, D_MLP = 113, 128, 4, 32, 512
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 12000
EVERY = 100


class OneLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        r = lambda *shape: torch.nn.Parameter(torch.randn(*shape) / math.sqrt(D))
        self.W_E, self.W_pos = r(P + 1, D), r(3, D)
        self.W_Q, self.W_K, self.W_V, self.W_O = r(HEADS, D_HEAD, D), r(HEADS, D_HEAD, D), r(HEADS, D_HEAD, D), r(HEADS, D, D_HEAD)
        self.W_in, self.W_out = r(D_MLP, D), r(D, D_MLP)
        self.b_in, self.b_out = torch.nn.Parameter(torch.zeros(D_MLP)), torch.nn.Parameter(torch.zeros(D))
        self.W_U = torch.nn.Parameter(torch.randn(D, P) / math.sqrt(P))

    def forward(self, tokens):
        """Logits at the last position. With one layer, nothing else influences them."""
        x = self.W_E[tokens] + self.W_pos                          # [batch, 3, d]
        query = torch.einsum("hed,bd->bhe", self.W_Q, x[:, -1])     # the last position reads
        key = torch.einsum("hed,bpd->bhpe", self.W_K, x)
        value = torch.einsum("hed,bpd->bhpe", self.W_V, x)
        pattern = (torch.einsum("bhe,bhpe->bhp", query, key) / math.sqrt(D_HEAD)).softmax(-1)
        z = torch.einsum("bhp,bhpe->bhe", pattern, value)
        mid = x[:, -1] + torch.einsum("hde,bhe->bd", self.W_O, z)
        neurons = torch.relu(mid @ self.W_in.T + self.b_in)
        return (mid + neurons @ self.W_out.T + self.b_out) @ self.W_U


pairs = torch.cartesian_prod(torch.arange(P), torch.arange(P))
tokens = torch.cat([pairs, torch.full((P * P, 1), P)], dim=1)       # a, b, "="
labels = (pairs[:, 0] + pairs[:, 1]) % P
perm = torch.randperm(P * P)
train, test = perm[: int(0.3 * P * P)], perm[int(0.3 * P * P):]

model = OneLayer()
CHECKPOINT = ROOT / "code" / "modadd_checkpoint.pt"
start = time.time()
if "--replot" in sys.argv and CHECKPOINT.exists():                     # reuse a finished run, only redo the analysis
    saved = torch.load(CHECKPOINT)
    model.load_state_dict(saved["model"]); history = saved["history"]; snapshots = list(saved["snapshots"])
    EPOCHS = history["epoch"][-1]
else:
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    loss_of = lambda idx: F.cross_entropy(model(tokens[idx]).double(), labels[idx])
    history = {"epoch": [], "train_loss": [], "test_loss": [], "test_acc": []}
    snapshots = []
    for epoch in range(EPOCHS + 1):
        loss = loss_of(train)
        if epoch % EVERY == 0:
            with torch.no_grad():
                logits = model(tokens[test]).double()
                history["epoch"].append(epoch)
                history["train_loss"].append(loss.item())
                history["test_loss"].append(F.cross_entropy(logits, labels[test]).item())
                history["test_acc"].append((logits.argmax(-1) == labels[test]).float().mean().item())
                snapshots.append(model.W_E[:P].detach().clone())
            if epoch % 1000 == 0:
                print(f"epoch {epoch:6d}  train {history['train_loss'][-1]:.2e}  test {history['test_loss'][-1]:.2e}  "
                      f"acc {history['test_acc'][-1]:.3f}  ({time.time() - start:.0f}s)", flush=True)
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    torch.save({"model": model.state_dict(), "history": history, "snapshots": torch.stack(snapshots),
                "seconds": time.time() - start}, CHECKPOINT)

# ---------------------------------------------------------------------------------------------------
# Fourier analysis of the embedding matrix: energy of each frequency k = 1, ..., 56.
# ---------------------------------------------------------------------------------------------------
n = torch.arange(P).double()
freqs = np.arange(1, P // 2 + 1)
basis = [torch.ones(P).double() / math.sqrt(P)]
for k in freqs:
    basis += [torch.cos(2 * math.pi * k * n / P) * math.sqrt(2 / P), torch.sin(2 * math.pi * k * n / P) * math.sqrt(2 / P)]
basis = torch.stack(basis)                                            # orthonormal, [113, 113]


def energy(W_E):
    coeff = (basis @ W_E.double()).pow(2).sum(-1)                     # squared norm of each Fourier coefficient
    per_freq = coeff[1::2] + coeff[2::2]
    return (per_freq / coeff.sum()).numpy()


final_energy = energy(snapshots[-1])
key = [int(k) for k in freqs[final_energy > 0.05]]
if not key:                                                           # not grokked yet: fall back to the largest five
    key = sorted(int(k) for k in freqs[np.argsort(-final_energy)[:5]])
key_index = np.array(key, dtype=int) - 1
key_share = [float(energy(W)[key_index].sum()) for W in snapshots]

# How well does the sum of cosines describe the logits?
with torch.no_grad():
    logits = model(tokens).double().reshape(P, P, P)
    centred = logits - logits.mean(-1, keepdim=True)
    a, b, c = torch.meshgrid(torch.arange(P), torch.arange(P), torch.arange(P), indexing="ij")
    diff = ((a + b - c) % P).double()
    design = torch.stack([torch.cos(2 * math.pi * k * diff / P).flatten() for k in key], dim=1)
    coef = torch.linalg.lstsq(design, centred.flatten().unsqueeze(1)).solution.squeeze(1)
    explained = 1 - ((centred.flatten() - design @ coef).pow(2).sum() / centred.pow(2).sum()).item()
    by_diff = torch.stack([centred.flatten()[diff.flatten() == m].mean() for m in range(P)]).numpy()
    fit_by_diff = sum(coef[i].item() * np.cos(2 * math.pi * k * np.arange(P) / P) for i, k in enumerate(key))

grok = next((e for e, acc in zip(history["epoch"], history["test_acc"]) if acc > 0.99), None)
results = {"history": history, "key_frequencies": key, "energy_per_frequency": final_energy.tolist(),
           "key_share_over_training": key_share, "cosine_fit_coefficients": coef.tolist(),
           "cosine_fit_variance_explained": explained, "first_epoch_test_acc_above_0.99": grok,
           "final_train_loss": history["train_loss"][-1], "final_test_loss": history["test_loss"][-1],
           "final_test_acc": history["test_acc"][-1], "epochs": EPOCHS,
           "key_share_final": key_share[-1], "weighted_fit_at_correct_answer": float(fit_by_diff[0]),
           "weighted_fit_largest_wrong_answer": float(fit_by_diff[1:].max())}
(ROOT / "code" / "results_modadd.json").write_text(json.dumps(results, indent=1))

# ---------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
ax = axes[0]
ax.semilogy(history["epoch"], history["train_loss"], color=BLUE, lw=1.6, label="training pairs")
ax.semilogy(history["epoch"], history["test_loss"], color=RED, lw=1.6, label="held-out pairs")
ax.set_xlabel("training step"); ax.set_ylabel("loss"); ax.legend(loc="upper right", fontsize=8); grid(ax)
ax = axes[1]
ax.plot(history["epoch"], key_share, color=BLUE, lw=1.6)
ax.set_xlabel("training step"); ax.set_ylabel("share of key frequencies in $W_E$"); ax.set_ylim(0, 1.02); grid(ax)
fig.tight_layout(w_pad=2.0); fig.savefig(FIG / "fig_grokking.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.3, 2.1))
colours = [RED if k in key else MUTED for k in freqs]
ax.bar(freqs, final_energy, color=colours, width=0.7)
for k in key:
    ax.text(k, final_energy[k - 1] + 0.01, str(k), ha="center", fontsize=8, color=INK)
ax.set_xlabel("frequency $r$"); ax.set_ylabel("share of $\\|W_E\\|^2$"); grid(ax)
fig.tight_layout(); fig.savefig(FIG / "fig_fourier.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.3, 2.3))
m = np.arange(P); shifted = (m + P // 2) % P - P // 2; order = np.argsort(shifted)
ax.plot(shifted[order], by_diff[order], color=BLUE, lw=0, marker="o", ms=2.6, label="trained network (mean logit)")
ax.plot(shifted[order], fit_by_diff[order], color=RED, lw=1.0, label="weighted sum of cosines")
ax.set_xlabel("$a+b-c$ modulo 113"); ax.set_ylabel("logit of $c$"); ax.legend(loc="upper right", fontsize=8); grid(ax)
fig.tight_layout(); fig.savefig(FIG / "fig_modadd_logits.pdf"); plt.close(fig)
print(json.dumps({k: v for k, v in results.items() if k not in ("history", "energy_per_frequency", "key_share_over_training")}, indent=1))
