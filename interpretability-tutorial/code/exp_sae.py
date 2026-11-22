"""Session 5: a pretrained sparse autoencoder on the residual stream of GPT-2 small (input of layer 8).

The autoencoder is the public one of Bloom (2024), 24,576 latents, downloaded from the Hugging Face Hub.
Writes code/results_sae.json and figures/fig_sae_stats.pdf, fig_sae_cosines.pdf. About three minutes on a CPU.
"""
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from common import load_model, ioi_prompts, MALE, FEMALE, GENDER_TEMPLATE, NAME_POS
from plotstyle import plt, BLUE, RED, MUTED, grid

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
torch.manual_seed(0)
model = load_model()
LAYER, HOOK = 8, "blocks.8.hook_resid_pre"
weights = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"{HOOK}/sae_weights.safetensors"))
W_enc, W_dec, b_enc, b_dec = weights["W_enc"], weights["W_dec"], weights["b_enc"], weights["b_dec"]
n_latents = W_dec.shape[0]
encode = lambda x: torch.relu((x - b_dec) @ W_enc + b_enc)
decode = lambda f: f @ W_dec + b_dec

# ---- Text: 300 documents of 128 tokens --------------------------------------------------------------
N_DOCS, CONTEXT, BATCH, KEEP = 300, 128, 25, 8
docs = []
for row in load_dataset("NeelNanda/pile-10k", split="train"):
    ids = model.to_tokens(row["text"])[0, :CONTEXT]
    if ids.shape[0] == CONTEXT:
        docs.append(ids)
    if len(docs) == N_DOCS:
        break
tokens = torch.stack(docs)


def next_token_loss(logits, batch):
    return torch.nn.functional.cross_entropy(logits[:, 1:-1].flatten(0, 1), batch[:, 2:].flatten())   # skip the start token


sq_err = sq_tot = n_active = n_tokens = norm_ratio = 0.0
losses = {"clean": [], "spliced": [], "zero": []}
fired = torch.zeros(n_latents)
top_values = torch.zeros(KEEP, n_latents); top_index = torch.zeros(KEEP, n_latents, dtype=torch.long)
active_per_token = []
mean_stream = None
for start in range(0, N_DOCS, BATCH):
    batch = tokens[start:start + BATCH]
    store = {}
    def keep(resid, hook):
        store["x"] = resid.detach().clone(); return resid
    def splice(resid, hook):
        resid[:, 1:] = decode(encode(resid[:, 1:])); return resid
    def zero(resid, hook):
        resid[:, 1:] = 0.0; return resid
    with torch.no_grad():
        losses["clean"].append(next_token_loss(model.run_with_hooks(batch, fwd_hooks=[(HOOK, keep)]), batch).item())
        losses["spliced"].append(next_token_loss(model.run_with_hooks(batch, fwd_hooks=[(HOOK, splice)]), batch).item())
        losses["zero"].append(next_token_loss(model.run_with_hooks(batch, fwd_hooks=[(HOOK, zero)]), batch).item())
        x = store["x"][:, 1:].flatten(0, 1)                                    # all positions except the start token
        f = encode(x)
        sq_err += (x - decode(f)).pow(2).sum().item()
        centre = lambda y: y - b_dec                                           # lengths are measured from the decoder bias
        norm_ratio += (centre(decode(f)).norm(dim=-1) / centre(x).norm(dim=-1)).sum().item()
        mean_stream = x.mean(0) if mean_stream is None else mean_stream
        sq_tot += (x - mean_stream).pow(2).sum().item()
        active = (f > 0)
        n_active += active.sum().item(); n_tokens += x.shape[0]
        fired += active.sum(0)
        active_per_token.append(active.sum(1))
        values, index = f.topk(KEEP, dim=0)                                    # strongest tokens of this batch, per latent
        flat = index + start * (CONTEXT - 1)                                   # index into all (document, position >= 1)
        merged_values = torch.cat([top_values, values]); merged_index = torch.cat([top_index, flat])
        top_values, order = merged_values.topk(KEEP, dim=0)
        top_index = merged_index.gather(0, order)
frequency = fired / n_tokens
mean_loss = {k: float(np.mean(v)) for k, v in losses.items()}
results = {
    "layer": LAYER, "n_latents": n_latents, "n_tokens": int(n_tokens),
    "explained_variance": 1 - sq_err / sq_tot, "mean_active_latents_per_token": n_active / n_tokens,
    "mean_ratio_of_reconstruction_norm_to_activation_norm": norm_ratio / n_tokens,
    "loss": mean_loss, "loss_recovered": (mean_loss["zero"] - mean_loss["spliced"]) / (mean_loss["zero"] - mean_loss["clean"]),
    "share_of_latents_that_never_fired": float((fired == 0).float().mean()),
    "share_of_latents_firing_on_more_than_1_percent": float((frequency > 0.01).float().mean()),
    "median_share_of_tokens_on_which_a_latent_fires": float(frequency.median()),
}


def context(flat_index, width=4):
    doc, pos = divmod(int(flat_index), CONTEXT - 1)
    pos += 1
    before = model.to_string(tokens[doc, max(1, pos - width):pos])
    return (before + "[" + model.to_string(tokens[doc, pos:pos + 1]) + "]").replace("\n", " ")


W_U = model.W_U - model.W_U.mean(0, keepdim=True)                                   # centring as in the final LayerNorm
def describe(j):
    promoted = (W_dec[j] @ W_U).topk(6).indices
    return {"latent": int(j), "frequency": round(frequency[j].item(), 5), "max_activation": round(top_values[0, j].item(), 2),
            "top_contexts": [context(i) for i in top_index[:, j]], "promoted_tokens": [model.to_string(t) for t in promoted]}


# candidates for the table of the text: latents that fire on one token, and latents that fire on many tokens
eligible = (frequency > 2e-4) & (frequency < 5e-3) & (top_values[-1] > 0)
strings = lambda j: [model.to_string(tokens.flatten()[0:0]) for _ in ()]            # placeholder, replaced below
token_at = lambda i: tokens[divmod(int(i), CONTEXT - 1)[0], divmod(int(i), CONTEXT - 1)[1] + 1].item()
distinct = torch.tensor([len({token_at(i) for i in top_index[:, j]}) if eligible[j] else -1 for j in range(n_latents)])
single_token = [int(j) for j in torch.where(distinct == 1)[0][:400]]
many_tokens = [int(j) for j in torch.where(distinct >= 7)[0][:400]]
pick = lambda group, k: sorted(group, key=lambda j: -top_values[0, j].item())[:k]
results["candidates_single_token"] = [describe(j) for j in pick(single_token, 6)]
results["candidates_many_tokens"] = [describe(j) for j in pick(many_tokens, 14)]

# ---- The running example: which latents are active on the name of the indirect object? -----------------
clean, _ = ioi_prompts()
with torch.no_grad():
    _, cache = model.run_with_cache(model.to_tokens(clean))
f_io = encode(cache[HOOK][:, 2])                                                    # position of the indirect object
strongest = f_io.mean(0).topk(4)
results["ioi_latents_on_indirect_object"] = [dict(describe(j), mean_activation=round(v.item(), 2),
                                                  active_on_prompts=int((f_io[:, j] > 0).sum()))
                                             for v, j in zip(strongest.values, strongest.indices)]
results["ioi_active_latents_per_position"] = float((encode(cache[HOOK][:, 1:]) > 0).sum(-1).float().mean())

# ---- Six latents that are discussed in the text ---------------------------------------------------------
SHOWCASE = {5238: "after the prefix cross-", 614: "WikiLeaks", 824: "livestock", 1088: "adjectives that take a suffix",
            658: "first names", 18206: "female first names"}
results["showcase"] = [dict(describe(j), label=label) for j, label in SHOWCASE.items()]

# ---- Latents and the gender of a name: four candidates, tested on the prompts of session 3 -------------
NAME_LATENT, FEMALE_LATENT, LOOKALIKE, ALIGNED = 658, 18206, 13484, 11952
he, she = model.to_single_token(" he"), model.to_single_token(" she")
prompt_tokens = lambda names: model.to_tokens([GENDER_TEMPLATE.format(n) for n in names])
metric = lambda logits: (logits[:, -1, she] - logits[:, -1, he]).mean().item()
with torch.no_grad():
    logits_m, cache_m = model.run_with_cache(prompt_tokens(MALE))
    logits_f, cache_f = model.run_with_cache(prompt_tokens(FEMALE))
x_m, x_f = cache_m[HOOK][:, NAME_POS], cache_f[HOOK][:, NAME_POS]
gender = x_f.mean(0) - x_m.mean(0)                                                 # difference of means at layer 8
f_m, f_f = encode(x_m), encode(x_f)
cosine = lambda u, w: torch.nn.functional.cosine_similarity(u, w, dim=0).item()
candidates = {}
for label, j in (("first_names", NAME_LATENT), ("female_first_names", FEMALE_LATENT),
                 ("lookalike_from_top_examples", LOOKALIKE), ("most_aligned_direction", ALIGNED)):
    candidates[label] = dict(describe(j), fires_on_female_names=float((f_f[:, j] > 0).float().mean()),
                             fires_on_male_names=float((f_m[:, j] > 0).float().mean()),
                             mean_activation_on_female_names=f_f[:, j].mean().item(),
                             mean_activation_on_male_names=f_m[:, j].mean().item(),
                             cosine_with_gender_direction=cosine(gender, W_dec[j]))


def edit_and_measure(edit):
    def hook(resid, hook):
        resid[:, NAME_POS] = edit(resid[:, NAME_POS]); return resid
    with torch.no_grad():
        return metric(model.run_with_hooks(prompt_tokens(MALE), fwd_hooks=[(HOOK, hook)]))


add = lambda x, js: x + sum(f_f[:, j].mean() * W_dec[j] for j in js)               # switch latents on at their usual strength
most_female = (f_f.mean(0) - f_m.mean(0)).topk(3).indices.tolist()
others = [j for j in torch.randperm(n_latents)[:20].tolist() if j not in most_female][:10]
results["gender_and_latents"] = {
    "candidates": candidates,
    "three_latents_with_largest_female_minus_male_activation": most_female,
    "norm_of_gender_direction": gender.norm().item(),
    "norm_of_female_latent_at_usual_strength": (f_f[:, FEMALE_LATENT].mean() * W_dec[FEMALE_LATENT]).norm().item(),
    "logit_diff_male_names": metric(logits_m), "logit_diff_female_names": metric(logits_f),
    "male_names_plus_female_latent": edit_and_measure(lambda x: add(x, [FEMALE_LATENT])),
    "male_names_plus_three_latents": edit_and_measure(lambda x: add(x, most_female)),
    "male_names_plus_difference_of_means": edit_and_measure(lambda x: x + gender),
    "male_names_plus_random_latent": float(np.mean([edit_and_measure(lambda x, j=j: x + f_f[:, FEMALE_LATENT].mean() * W_dec[j]) for j in others])),
}

# ---- Steering text with one latent: add its direction at every position while the model writes ---------
def generate(prompt, latent=None, strength=0.0, n_new=24):
    ids = model.to_tokens(prompt)
    def hook(resid, hook):
        resid[:, 1:] += strength * W_dec[latent]; return resid
    for _ in range(n_new):
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=[(HOOK, hook)]) if latent is not None else model(ids)
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)       # greedy decoding
    return model.to_string(ids[0, 1:])


PROMPT = "I went for a walk and"
results["steered_text"] = {"prompt": PROMPT, "max_activation_of_wikileaks_latent": top_values[0, 614].item(),
                           "no_steering": generate(PROMPT),
                           **{f"wikileaks_latent_strength_{s}": generate(PROMPT, 614, float(s)) for s in (30, 60, 80, 120)},
                           "livestock_latent_strength_100": generate(PROMPT, 824, 100.0)}

# ---- Geometry of the dictionary -----------------------------------------------------------------------
sample = torch.nn.functional.normalize(W_dec[torch.randperm(n_latents)[:2000]], dim=-1)
cos_dict = (sample @ sample.T)[np.triu_indices(2000, k=1)].numpy()
random = torch.nn.functional.normalize(torch.randn(2000, 768), dim=-1)
cos_rand = (random @ random.T)[np.triu_indices(2000, k=1)].numpy()
results["dictionary_cosines"] = {"std": float(cos_dict.std()), "max_abs": float(np.abs(cos_dict).max()),
                                 "share_above_0.1": float((np.abs(cos_dict) > 0.1).mean()),
                                 "share_above_0.3": float((np.abs(cos_dict) > 0.3).mean()),
                                 "decoder_norm_mean": float(W_dec.norm(dim=-1).mean())}
torch.save({"frequency": frequency, "top_values": top_values, "top_index": top_index, "tokens": tokens},
           ROOT / "code" / "sae_cache.pt")
(ROOT / "code" / "results_sae.json").write_text(json.dumps(results, indent=1))

# ---------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
ax = axes[0]
ax.hist(torch.cat(active_per_token).numpy(), bins=60, color=BLUE)
ax.set_xlabel("active latents per token (of 24,576)"); ax.set_ylabel("number of tokens"); grid(ax)
ax = axes[1]
ax.hist(np.log10(frequency[frequency > 0].numpy()), bins=60, color=BLUE)
ax.set_xlabel("$\\log_{10}$ (share of tokens with the latent active)"); ax.set_ylabel("number of latents"); grid(ax)
fig.tight_layout(w_pad=2.0); fig.savefig(FIG / "fig_sae_stats.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 2.4))
bins = np.linspace(-0.3, 0.3, 121)
ax.hist(cos_rand, bins=bins, histtype="step", color=MUTED, lw=1.2, label="random unit vectors")
ax.hist(cos_dict, bins=bins, histtype="step", color=BLUE, lw=1.4, label="dictionary directions")
ax.set_yscale("log"); ax.set_xlabel("inner product of two directions"); ax.set_ylabel("number of pairs"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_sae_cosines.pdf"); plt.close(fig)

show = {k: v for k, v in results.items() if not k.startswith("candidates") and not k.startswith("ioi_latents") and k != "showcase"}
print(json.dumps(show, indent=0))
for group in ("candidates_single_token", "candidates_many_tokens", "ioi_latents_on_indirect_object"):
    print("==", group)
    for c in results[group]:
        print(c["latent"], c["frequency"], c["max_activation"], "|", " ~ ".join(c["top_contexts"][:6])[:330], "| promotes", c["promoted_tokens"])
