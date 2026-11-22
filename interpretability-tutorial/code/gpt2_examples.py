"""Worked examples for the tutorial, all on GPT-2 small (about four minutes on a 4-core laptop CPU).

Produces the numbers in code/results.json and the figures in figures/:
  fig_logit_lens.pdf      logit lens on the indirect-object task
  fig_dla_heads.pdf       direct logit attribution of every attention head
  fig_resid_patching.pdf  activation patching of the residual stream (layer x position)
  fig_attr_vs_act.pdf     attribution patching versus activation patching, per head

Tested with transformer_lens 4.4.0, torch 2.15, Python 3.13.
Usage:  python code/gpt2_examples.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from transformer_lens import TransformerBridge

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
torch.manual_seed(0)

# TransformerLens v4: TransformerBridge wraps the Hugging Face model. Compatibility mode folds the
# LayerNorm weights into the next linear map, so every LayerNorm is x -> (x - mean(x)) / scale(x).
model = TransformerBridge.boot_transformers("gpt2")
model.enable_compatibility_mode()
n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

# ----------------------------------------------------------------------------------------------
# Data: one template, six name pairs, both orders. The clean prompt ends in the repeated name B,
# so the answer is A. The corrupted prompt repeats A instead, so its answer is B.
# ----------------------------------------------------------------------------------------------
PAIRS = [(" Mary", " John"), (" Tom", " James"), (" Dan", " Sarah"),
         (" Anna", " Mark"), (" Kate", " Paul"), (" Lisa", " David")]
PAIRS = PAIRS + [(b, a) for a, b in PAIRS]
TEMPLATE = "When{A} and{B} went to the store,{S} gave a drink to"

clean_prompts = [TEMPLATE.format(A=a, B=b, S=b) for a, b in PAIRS]
corrupt_prompts = [TEMPLATE.format(A=a, B=b, S=a) for a, b in PAIRS]
clean_tokens = model.to_tokens(clean_prompts)
corrupt_tokens = model.to_tokens(corrupt_prompts)
assert clean_tokens.shape == corrupt_tokens.shape
answer = torch.tensor([model.to_single_token(a) for a, _ in PAIRS])   # correct on the clean prompt
wrong = torch.tensor([model.to_single_token(b) for _, b in PAIRS])    # the repeated name
n_prompts, n_pos = clean_tokens.shape
rows = torch.arange(n_prompts)
pos_labels = ["BOS", "When", "IO", "and", "S1", "went", "to", "the", "store", ",", "S2", "gave", "a", "drink", "to"]
assert len(pos_labels) == n_pos


def logit_diff(logits):
    """Metric: logit(correct name) - logit(repeated name) at the last position, averaged over prompts."""
    last = logits[:, -1, :]
    return (last[rows, answer] - last[rows, wrong]).mean()


with torch.no_grad():
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)
ld_clean, ld_corrupt = logit_diff(clean_logits).item(), logit_diff(corrupt_logits).item()
probs = clean_logits[:, -1, :].softmax(-1)
results = {
    "n_prompts": n_prompts,
    "example_clean_prompt": clean_prompts[0],
    "example_corrupt_prompt": corrupt_prompts[0],
    "logit_diff_clean": ld_clean,
    "logit_diff_corrupt": ld_corrupt,
    "per_prompt_logit_diff_clean": (clean_logits[rows, -1, answer] - clean_logits[rows, -1, wrong]).tolist(),
    "mean_prob_correct_clean": probs[rows, answer].mean().item(),
    "mean_prob_repeated_clean": probs[rows, wrong].mean().item(),
    "top1_is_correct_fraction": (probs.argmax(-1) == answer).float().mean().item(),
}


def normalise(metric_value):
    """0 = corrupted behaviour, 1 = clean behaviour."""
    return (metric_value - ld_corrupt) / (ld_clean - ld_corrupt)


# ----------------------------------------------------------------------------------------------
# 0. A check quoted in an exercise: GPT-2 small does not reliably know where the Eiffel Tower is.
# ----------------------------------------------------------------------------------------------
with torch.no_grad():
    fact_logits = model(model.to_tokens("The Eiffel Tower is located in the city of"))
fact_top = fact_logits[0, -1].softmax(-1).topk(5)
results["eiffel_tower_top5"] = [(model.to_string(i), v.item()) for v, i in zip(fact_top.values, fact_top.indices)]

# ----------------------------------------------------------------------------------------------
# 0a. Attention patterns: where do a few late heads look from the last position?
# ----------------------------------------------------------------------------------------------
attention = {}
for name in ("9.9", "9.6", "10.0", "10.7", "11.10"):
    layer, head = map(int, name.split("."))
    pattern = clean_cache[f"blocks.{layer}.attn.hook_pattern"][:, head, -1, :]     # [prompt, source position]
    attention[name] = {label: pattern[:, pos].mean().item() for label, pos in (("IO", 2), ("S1", 4), ("S2", 10))}
results["attention_from_last_position"] = attention

# ----------------------------------------------------------------------------------------------
# 0b. Input attribution. F maps the token embeddings to the logit difference. The baseline replaces
#     every token embedding by zero (positional embeddings are kept). Integrated gradients uses a
#     midpoint Riemann sum with n_steps points along the straight line from the baseline to the input.
# ----------------------------------------------------------------------------------------------
embeddings = clean_cache["hook_embed"].detach()                                   # [prompt, position, d]


def metric_and_gradient(fraction):
    """Value and gradient of the mean logit difference at the point baseline + fraction * (input - baseline)."""
    kept = {}
    def replace(embed, hook):
        point = (fraction * embeddings).clone().requires_grad_(True)
        kept["point"] = point
        return point
    value = logit_diff(model.run_with_hooks(clean_tokens, fwd_hooks=[("hook_embed", replace)]))
    value.backward()
    return value.item(), kept["point"].grad.detach()


n_steps = 64
f_input, grad_at_input = metric_and_gradient(1.0)
f_baseline, _ = metric_and_gradient(0.0)
average_gradient = sum(metric_and_gradient((m + 0.5) / n_steps)[1] for m in range(n_steps)) / n_steps
# the metric is a mean over prompts, so sums over prompts of gradient * input are means of per-prompt attributions
integrated_gradients = (average_gradient * embeddings).sum(dim=(0, 2))            # [position]
gradient_times_input = (grad_at_input * embeddings).sum(dim=(0, 2))
results["input_attribution"] = {
    "positions": pos_labels,
    "integrated_gradients": integrated_gradients.tolist(),
    "gradient_times_input": gradient_times_input.tolist(),
    "sum_integrated_gradients": integrated_gradients.sum().item(),
    "sum_gradient_times_input": gradient_times_input.sum().item(),
    "F_input_minus_F_baseline": f_input - f_baseline,
    "F_input": f_input, "F_baseline": f_baseline, "n_steps": n_steps,
}

# ----------------------------------------------------------------------------------------------
# 1. Logit lens: decode the residual stream after each layer with the final LayerNorm + unembedding.
# ----------------------------------------------------------------------------------------------
lens_ld, lens_prob, lens_top1 = [], [], []
with torch.no_grad():
    for layer in range(n_layers + 1):
        name = f"blocks.{layer}.hook_resid_pre" if layer < n_layers else f"blocks.{n_layers - 1}.hook_resid_post"
        resid_last = clean_cache[name][:, -1:, :]
        lens_logits = model.unembed(model.ln_final(resid_last))
        lens_ld.append(logit_diff(lens_logits).item())
        lens_prob.append(lens_logits[:, -1, :].softmax(-1)[rows, answer].mean().item())
        lens_top1.append(model.to_string(lens_logits[0, -1].argmax()))
results["logit_lens"] = {"logit_diff": lens_ld, "prob_correct": lens_prob, "top1_first_prompt": lens_top1}

# ----------------------------------------------------------------------------------------------
# 2. Direct logit attribution. With the final LayerNorm scale frozen at its value on this input,
#    logit_diff = sum over components c of  ((c - mean(c)) / scale) . (W_U[:, answer] - W_U[:, wrong]).
# ----------------------------------------------------------------------------------------------
with torch.no_grad():
    W_U, b_U = model.W_U, model.b_U                                     # [d, vocab], [vocab]
    direction = (W_U[:, answer] - W_U[:, wrong]).T                      # [prompt, d]
    scale = clean_cache["ln_final.hook_scale"][:, -1, :]                # [prompt, 1]

    def dla(component):                                                 # component: [prompt, d]
        centred = component - component.mean(-1, keepdim=True)
        return ((centred / scale) * direction).sum(-1).mean().item()

    head_dla = np.zeros((n_layers, n_heads))
    head_dla_corrupt = np.zeros((n_layers, n_heads))   # same read-out applied to the head's corrupted output
    mlp_dla, attn_bias_dla = [], []
    for layer in range(n_layers):
        z = clean_cache[f"blocks.{layer}.attn.hook_z"][:, -1, :, :]     # [prompt, head, d_head]
        for head in range(n_heads):
            head_dla[layer, head] = dla(z[:, head, :] @ model.W_O[layer, head])
            z_corrupt = corrupt_cache[f"blocks.{layer}.attn.hook_z"][:, -1, head, :]
            head_dla_corrupt[layer, head] = dla(z_corrupt @ model.W_O[layer, head])
        attn_bias_dla.append(dla(model.b_O[layer].expand(n_prompts, -1)))
        mlp_dla.append(dla(clean_cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :]))
    embed_dla = dla(clean_cache["hook_embed"][:, -1, :]) + dla(clean_cache["hook_pos_embed"][:, -1, :])
    bias_term = (b_U[answer] - b_U[wrong]).mean().item()
    total = head_dla.sum() + sum(mlp_dla) + sum(attn_bias_dla) + embed_dla + bias_term
top = sorted(((head_dla[l, h], f"{l}.{h}") for l in range(n_layers) for h in range(n_heads)), reverse=True)
results["dla"] = {
    "sum_of_all_components": float(total),
    "actual_logit_diff": ld_clean,
    "heads_total": float(head_dla.sum()), "mlps_total": float(sum(mlp_dla)),
    "embed_total": float(embed_dla), "attn_bias_total": float(sum(attn_bias_dla)), "unembed_bias": bias_term,
    "mlp_by_layer": mlp_dla,
    "top_positive_heads": [(name, float(v)) for v, name in top[:6]],
    "top_negative_heads": [(name, float(v)) for v, name in top[-4:]],
}

# ----------------------------------------------------------------------------------------------
# 3. Activation patching of the residual stream (denoising): run the corrupted prompt, overwrite
#    one (layer, position) with its value from the clean run, and measure how much is restored.
# ----------------------------------------------------------------------------------------------
resid_patch = np.zeros((n_layers, n_pos))
with torch.no_grad():
    for layer in range(n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        for pos in range(n_pos):
            def patch(resid, hook):
                resid[:, pos, :] = clean_cache[hook.name][:, pos, :]
                return resid
            patched_logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(name, patch)])
            resid_patch[layer, pos] = normalise(logit_diff(patched_logits).item())
results["resid_patching"] = resid_patch.tolist()

# ----------------------------------------------------------------------------------------------
# 4. Per-head patching (exact) versus attribution patching (first-order approximation).
#    Node = the output of one head (hook_z) at all positions.
# ----------------------------------------------------------------------------------------------
head_denoise = np.zeros((n_layers, n_heads))   # corrupted run, clean head patched in
head_noise = np.zeros((n_layers, n_heads))     # clean run, corrupted head patched in
with torch.no_grad():
    for layer in range(n_layers):
        name = f"blocks.{layer}.attn.hook_z"
        for head in range(n_heads):
            def patch_clean(z, hook):
                z[:, :, head, :] = clean_cache[hook.name][:, :, head, :]
                return z
            def patch_corrupt(z, hook):
                z[:, :, head, :] = corrupt_cache[hook.name][:, :, head, :]
                return z
            out = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(name, patch_clean)])
            head_denoise[layer, head] = normalise(logit_diff(out).item())
            out = model.run_with_hooks(clean_tokens, fwd_hooks=[(name, patch_corrupt)])
            head_noise[layer, head] = normalise(logit_diff(out).item())

# Attribution patching: one forward and one backward pass on the corrupted prompt give the gradient
# of the metric with respect to every head output at once.
stored = {}
def keep(z, hook):
    z.retain_grad()
    stored[hook.name] = z
    return z
hooks = [(f"blocks.{layer}.attn.hook_z", keep) for layer in range(n_layers)]
metric = logit_diff(model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks))
metric.backward()
head_attr = np.zeros((n_layers, n_heads))
for layer in range(n_layers):
    name = f"blocks.{layer}.attn.hook_z"
    delta = clean_cache[name] - corrupt_cache[name]                     # [prompt, pos, head, d_head]
    # the metric is a mean over prompts, so summing grad * delta over prompts gives the mean effect
    estimate = (stored[name].grad * delta).sum(dim=(0, 1, 3))
    head_attr[layer] = (estimate / (ld_clean - ld_corrupt)).detach().numpy()

flat = lambda a: [(f"{l}.{h}", float(a[l, h])) for l in range(n_layers) for h in range(n_heads)]
order = np.argsort(-np.abs(head_denoise).ravel())[:14]
results["head_patching"] = {
    "pearson_r_attr_vs_denoise": float(np.corrcoef(head_denoise.ravel(), head_attr.ravel())[0, 1]),
    "top_heads_by_abs_denoising_effect": [
        {"head": f"{i // n_heads}.{i % n_heads}", "denoise": float(head_denoise.ravel()[i]),
         "attribution": float(head_attr.ravel()[i]), "noise": float(head_noise.ravel()[i]),
         "dla": float(head_dla.ravel()[i]),
         "direct_effect_of_noising_logits": float((head_dla - head_dla_corrupt).ravel()[i]),
         "total_effect_of_noising_logits": float(ld_clean - (ld_corrupt + head_noise.ravel()[i] * (ld_clean - ld_corrupt)))}
        for i in order],
    "noise_all_logits_direct_vs_total": [
        (f"{l}.{h}", float(head_dla[l, h] - head_dla_corrupt[l, h]),
         float((1 - head_noise[l, h]) * (ld_clean - ld_corrupt))) for l in range(n_layers) for h in range(n_heads)],
    "denoise_all": flat(head_denoise), "noise_all": flat(head_noise), "attribution_all": flat(head_attr),
}
(ROOT / "code" / "results.json").write_text(json.dumps(results, indent=1))

# ----------------------------------------------------------------------------------------------
# Figures. Signed quantities use one diverging blue-grey-red map centred at zero.
# ----------------------------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "cmr10", "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#0f4c9c", "#2a78d6", "#f0efec", "#d9483b", "#9a1f17"])
BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"


def heatmap(ax, data, limit, label):
    rows_, cols_ = data.shape
    image = ax.pcolormesh(np.arange(cols_ + 1) - 0.5, np.arange(rows_ + 1) - 0.5, data, cmap=DIVERGING,
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit), edgecolors="white", linewidth=0.6)
    ax.invert_yaxis()
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    bar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.set_label(label)
    return image


# Figure: logit lens (two small panels, one measure each)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
layers = np.arange(n_layers + 1)
for ax, values, label in ((axes[0], lens_ld, "logit difference"), (axes[1], lens_prob, "probability of correct name")):
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.plot(layers, values, color=BLUE, lw=2, marker="o", ms=4)
    ax.set_xlabel("number of layers applied")
    ax.set_ylabel(label)
    ax.set_xticks(layers[::2])
    ax.grid(axis="y", color="#e4e3df", lw=0.5)
fig.tight_layout(w_pad=2.5)
fig.savefig(FIG / "fig_logit_lens.pdf")
plt.close(fig)

# Figure: direct logit attribution per head
fig, ax = plt.subplots(figsize=(4.3, 3.3))
heatmap(ax, head_dla, np.abs(head_dla).max(), "direct contribution to logit difference")
ax.set_xlabel("head")
ax.set_ylabel("layer")
ax.set_xticks(range(n_heads))
ax.set_yticks(range(n_layers))
for name in ("9.9", "9.6", "10.0", "10.7", "11.10"):                  # the heads discussed in the text
    layer, head = map(int, name.split("."))
    strong = abs(head_dla[layer, head]) > 0.5 * np.abs(head_dla).max()
    ax.text(head, layer, name, ha="center", va="center", fontsize=6.5, color="white" if strong else INK)
fig.tight_layout()
fig.savefig(FIG / "fig_dla_heads.pdf")
plt.close(fig)

# Figure: residual stream patching
fig, ax = plt.subplots(figsize=(6.3, 3.0))
heatmap(ax, resid_patch, max(1.0, np.abs(resid_patch).max()), "restored fraction of logit difference")
ax.set_xticks(range(n_pos))
ax.set_xticklabels(pos_labels, rotation=45, ha="right")
ax.set_yticks(range(n_layers))
ax.set_xlabel("position")
ax.set_ylabel("layer (input to block)")
fig.tight_layout()
fig.savefig(FIG / "fig_resid_patching.pdf")
plt.close(fig)

# Figure: attribution patching versus activation patching
fig, ax = plt.subplots(figsize=(3.4, 3.2))
low = min(head_denoise.min(), head_attr.min()) - 0.06
high = max(head_denoise.max(), head_attr.max()) + 0.12
ax.plot([low, high], [low, high], color=MUTED, lw=0.8, ls="--")
ax.scatter(head_denoise.ravel(), head_attr.ravel(), s=14, color=BLUE, edgecolor="white", linewidth=0.5)
for i in np.argsort(-np.abs(head_denoise).ravel())[:6]:
    name = f"{i // n_heads}.{i % n_heads}"
    ax.annotate(name, (head_denoise.ravel()[i], head_attr.ravel()[i]), textcoords="offset points",
                xytext=(-19, 3) if name == "9.9" else (5, -3), fontsize=7.5, color=INK)
ax.set_xlabel("activation patching (exact)")
ax.set_ylabel("attribution patching (linear estimate)")
ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_aspect("equal")
ax.grid(color="#e4e3df", lw=0.5)
fig.tight_layout()
fig.savefig(FIG / "fig_attr_vs_act.pdf")
plt.close(fig)

print(json.dumps({k: v for k, v in results.items() if k not in ("resid_patching", "head_patching")}, indent=1))
print(json.dumps(results["input_attribution"], indent=1))
print(json.dumps(results["head_patching"]["top_heads_by_abs_denoising_effect"], indent=1))
print("pearson r:", results["head_patching"]["pearson_r_attr_vs_denoise"])
