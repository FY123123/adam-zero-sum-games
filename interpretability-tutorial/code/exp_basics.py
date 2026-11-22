"""Sessions 1 and 2: what GPT-2 small predicts, how it cuts text into tokens, where its heads look.

Writes code/results_basics.json and figures/fig_attention.pdf. About half a minute on a CPU.
"""
import json
from pathlib import Path

import numpy as np
import torch

from common import load_model, ioi_prompts, IOI_POSITIONS
from plotstyle import plt, heatmap

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
model = load_model()
results = {}

# ---- Tokens ------------------------------------------------------------------------------------------
results["tokens"] = {text: model.to_str_tokens(text) for text in
                     ["Interpretability is fun.", "When Mary and John went to the store, John gave a drink to"]}

# ---- Next-token predictions --------------------------------------------------------------------------
prompts = ["One, two, three, four,", "When Mary and John went to the store, John gave a drink to",
           "The Eiffel Tower is located in the city of"]
predictions = {}
with torch.no_grad():
    for prompt in prompts:
        top = model(model.to_tokens(prompt))[0, -1].softmax(-1).topk(5)
        predictions[prompt] = [(model.to_string(i), round(v.item(), 3)) for v, i in zip(top.values, top.indices)]
results["predictions"] = predictions

# ---- Attention patterns on the indirect-object prompts ------------------------------------------------
clean, _ = ioi_prompts()
tokens = model.to_tokens(clean)
with torch.no_grad():
    _, cache = model.run_with_cache(tokens)
n_layers, n_heads, n_pos = model.cfg.n_layers, model.cfg.n_heads, tokens.shape[1]
patterns = torch.stack([cache[f"blocks.{l}.attn.hook_pattern"] for l in range(n_layers)])   # [layer, prompt, head, dest, source]
previous = torch.stack([patterns[..., p, p - 1] for p in range(1, n_pos)]).mean(dim=(0, 2))  # [layer, head]
best = previous.flatten().topk(5)
results["previous_token_heads"] = [(f"{i // n_heads}.{i % n_heads}", round(v.item(), 3)) for v, i in zip(best.values, best.indices)]
top_layer, top_head = divmod(best.indices[0].item(), n_heads)
rows = [(9, 9), (9, 6), (10, 0), (10, 7)]
last_rows = torch.stack([patterns[l, :, h, -1, :].mean(0) for l, h in rows]).numpy()             # mean over prompts
results["attention_from_last_position"] = {f"{l}.{h}": list(zip(IOI_POSITIONS, np.round(last_rows[i], 3).tolist()))
                                           for i, (l, h) in enumerate(rows)}
(ROOT / "code" / "results_basics.json").write_text(json.dumps(results, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1.0, 1.25]})
ax = axes[0]
heatmap(ax, patterns[top_layer, 0, top_head].numpy(), "attention weight", signed=False, limit=1.0, colorbar=False)
labels = [t.strip() if t.strip() else t for t in model.to_str_tokens(tokens[0])]
labels[0] = "BOS"
ax.set_xticks(range(n_pos)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticks(range(n_pos)); ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("source"); ax.set_ylabel("destination"); ax.set_title(f"head {top_layer}.{top_head}", fontsize=9)
ax = axes[1]
heatmap(ax, last_rows, "attention weight", signed=False, limit=1.0)
ax.set_xticks(range(n_pos)); ax.set_xticklabels(IOI_POSITIONS, rotation=90, fontsize=7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"head {l}.{h}" for l, h in rows], fontsize=8)
ax.set_xlabel("source"); ax.set_title("attention from the last position", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig_attention.pdf"); plt.close(fig)
print(json.dumps(results, indent=0)[:2600])
