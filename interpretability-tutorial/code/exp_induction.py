"""Session 7: induction heads in GPT-2 small, shown on sequences of random tokens that are repeated once.

Writes code/results_induction.json and figures/fig_induction_loss.pdf, fig_induction_heads.pdf.
About one minute on a CPU.
"""
import json
from pathlib import Path

import numpy as np
import torch

from common import load_model
from plotstyle import plt, BLUE, RED, MUTED, INK, grid, heatmap

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
torch.manual_seed(0)
model = load_model()
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

BATCH, LENGTH = 20, 40
bos = model.to_tokens("")[0, 0].item()
half = torch.randint(1000, 40000, (BATCH, LENGTH))
tokens = torch.cat([torch.full((BATCH, 1), bos), half, half], dim=1)          # BOS, 40 random tokens, the same 40 again
n_pos = tokens.shape[1]


def loss_by_position(logits):
    """Cross-entropy of the prediction of token p+1 made at position p, averaged over the batch."""
    log_probs = logits[:, :-1].log_softmax(-1)
    return -log_probs.gather(-1, tokens[:, 1:, None]).squeeze(-1).mean(0)


with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)
loss = loss_by_position(logits)
second = slice(LENGTH + 1, n_pos - 1)                                           # predictions made inside the second copy
results = {"batch": BATCH, "length": LENGTH, "loss_first_half": loss[1:LENGTH].mean().item(),
           "loss_second_half": loss[second].mean().item(), "uniform_guess": float(np.log(model.cfg.d_vocab)),
           "loss_by_position": loss.tolist()}

# ---- Scores of all heads -----------------------------------------------------------------------------
patterns = torch.stack([cache[f"blocks.{l}.attn.hook_pattern"] for l in range(n_layers)])   # [layer, batch, head, dest, source]
dest = torch.arange(LENGTH + 1, n_pos)                                          # positions in the second copy
induction = patterns[:, :, :, dest, dest - LENGTH + 1].mean(dim=(1, 3))         # attends to the token after the earlier copy
duplicate = patterns[:, :, :, dest, dest - LENGTH].mean(dim=(1, 3))             # attends to the earlier copy itself
all_dest = torch.arange(1, n_pos)
previous = patterns[:, :, :, all_dest, all_dest - 1].mean(dim=(1, 3))           # attends to the previous position
name = lambda i: f"{i // n_heads}.{i % n_heads}"
top = lambda score, k=6: [(name(i.item()), round(v.item(), 3)) for v, i in zip(*score.flatten().topk(k))]
results.update({"top_induction_heads": top(induction), "top_duplicate_token_heads": top(duplicate),
                "top_previous_token_heads": top(previous)})


# ---- Ablation: replace the output of some heads by its mean over all positions and prompts -----------
def ablated_loss(heads):
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    def make(heads_of_layer):
        def hook(z, hook):
            for head in heads_of_layer:
                z[:, :, head, :] = z[:, :, head, :].mean(dim=(0, 1), keepdim=True)
            return z
        return hook
    hooks = [(f"blocks.{layer}.attn.hook_z", make(hs)) for layer, hs in by_layer.items()]
    with torch.no_grad():
        return loss_by_position(model.run_with_hooks(tokens, fwd_hooks=hooks))[second].mean().item()


ranked = [divmod(i.item(), n_heads) for i in induction.flatten().argsort(descending=True)]
counts = [0, 1, 2, 3, 4, 5, 8]
ablate_induction = [ablated_loss(ranked[:k]) for k in counts]
generator = torch.Generator().manual_seed(1)
ablate_random = []
for k in counts:
    draws = []
    for _ in range(5):
        chosen = [divmod(i.item(), n_heads) for i in torch.randperm(n_layers * n_heads, generator=generator)[:k]]
        draws.append(ablated_loss(chosen))
    ablate_random.append(float(np.mean(draws)))
results.update({"ablation_counts": counts, "loss_second_half_ablating_induction_heads": ablate_induction,
                "loss_second_half_ablating_random_heads": ablate_random})
(ROOT / "code" / "results_induction.json").write_text(json.dumps(results, indent=1))

# ---------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
ax = axes[0]
ax.plot(np.arange(1, n_pos - 1), loss[1:].numpy(), color=BLUE, lw=1.6)
ax.axvline(LENGTH + 0.5, color=MUTED, lw=0.8, ls="--")
ax.text(LENGTH + 2, loss.max().item() * 0.92, "second copy starts", fontsize=8, color=MUTED)
ax.set_xlabel("position"); ax.set_ylabel("loss on the next token"); grid(ax)
ax = axes[1]
ax.plot(counts, ablate_induction, color=RED, lw=1.8, marker="o", ms=4, label="top induction heads")
ax.plot(counts, ablate_random, color=MUTED, lw=1.4, ls="--", marker="o", ms=3, label="random heads")
ax.set_xlabel("number of heads ablated"); ax.set_ylabel("loss in the second copy"); ax.legend(fontsize=8); grid(ax)
fig.tight_layout(w_pad=2.0); fig.savefig(FIG / "fig_induction_loss.pdf"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
ax = axes[0]
heatmap(ax, induction.numpy(), "induction score", signed=False, limit=1.0)
ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_xticks(range(n_heads)); ax.set_yticks(range(n_layers))
for layer, head in ranked[:5]:
    ax.text(head, layer, f"{layer}.{head}", ha="center", va="center", fontsize=6, color="white")
ax = axes[1]
layer, head = ranked[0]
ax.imshow(patterns[layer, 0, head].numpy(), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
ax.set_xlabel("source position"); ax.set_ylabel("destination position"); ax.set_title(f"attention of head {layer}.{head}", fontsize=9)
ax.set_xticks([0, LENGTH, 2 * LENGTH]); ax.set_yticks([0, LENGTH, 2 * LENGTH])
for side in ("top", "right"):
    ax.spines[side].set_visible(True)
fig.tight_layout(w_pad=1.5); fig.savefig(FIG / "fig_induction_heads.pdf"); plt.close(fig)
print(json.dumps({k: v for k, v in results.items() if k != "loss_by_position"}, indent=0))
