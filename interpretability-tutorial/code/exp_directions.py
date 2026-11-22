"""Session 3: a concept as a direction. Probing and steering the gender of a name in GPT-2 small.

Prompt: "After <name> got home," and the model has to choose between " he" and " she".
Writes code/results_directions.json and figures/fig_probe_layers.pdf, fig_steering.pdf. About two minutes.
"""
import json
from pathlib import Path

import numpy as np
import torch

from common import load_model, MALE, FEMALE, GENDER_TEMPLATE, NAME_POS
from plotstyle import plt, BLUE, RED, MUTED, grid

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
torch.manual_seed(0)
model = load_model()
n_layers = model.cfg.n_layers

TEMPLATE, LAST = GENDER_TEMPLATE, -1
he, she = model.to_single_token(" he"), model.to_single_token(" she")

order = torch.randperm(len(MALE)).tolist()
n_train = 36
split = lambda names: ([names[i] for i in order[:n_train]], [names[i] for i in order[n_train:]])
(male_train, male_test), (female_train, female_test) = split(MALE), split(FEMALE)
tokens_of = lambda names: model.to_tokens([TEMPLATE.format(n) for n in names])
metric = lambda logits: (logits[:, -1, she] - logits[:, -1, he]).mean().item()        # > 0 means "she"


def streams(names):
    """Residual stream at every layer, [layer 0..12, name, position, d]."""
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens_of(names))
    layers = [cache[f"blocks.{l}.hook_resid_pre"] for l in range(n_layers)] + [cache[f"blocks.{n_layers - 1}.hook_resid_post"]]
    return torch.stack(layers), logits


acts = {key: streams(names) for key, names in
        {"male_train": male_train, "male_test": male_test, "female_train": female_train, "female_test": female_test}.items()}
results = {"template": TEMPLATE, "n_names_per_gender": len(MALE), "n_train_per_gender": n_train,
           "baseline_logit_diff": {k: metric(v[1]) for k, v in acts.items()},
           "accuracy_of_the_model": float(np.mean([(acts["male_test"][1][:, -1, she] < acts["male_test"][1][:, -1, he]).float().mean().item(),
                                                     (acts["female_test"][1][:, -1, she] > acts["female_test"][1][:, -1, he]).float().mean().item()]))}


# ---- Linear probes ------------------------------------------------------------------------------------
def fit_probe(x, y, steps=300, weight_decay=1e-2):
    """Logistic regression on standardised inputs. Returns a function that predicts labels and the weight vector."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    w = torch.zeros(x.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([w, b], max_iter=steps)
    def closure():
        optimiser.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(((x - mean) / std) @ w + b, y) + weight_decay * w.pow(2).sum()
        loss.backward(); return loss
    optimiser.step(closure)
    return (lambda z: ((((z - mean) / std) @ w + b) > 0).float()), (w / std).detach()


def probe_accuracy(position, labels_train=None):
    accuracy = []
    for layer in range(n_layers + 1):
        x_train = torch.cat([acts["male_train"][0][layer, :, position], acts["female_train"][0][layer, :, position]])
        x_test = torch.cat([acts["male_test"][0][layer, :, position], acts["female_test"][0][layer, :, position]])
        y_train = torch.cat([torch.zeros(n_train), torch.ones(n_train)]) if labels_train is None else labels_train
        y_test = torch.cat([torch.zeros(len(male_test)), torch.ones(len(female_test))])
        predict, _ = fit_probe(x_train, y_train)
        accuracy.append((predict(x_test) == y_test).float().mean().item())
    return accuracy


results["probe_accuracy_name_position"] = probe_accuracy(NAME_POS)
results["probe_accuracy_last_position"] = probe_accuracy(LAST)
shuffled = [probe_accuracy(NAME_POS, torch.cat([torch.zeros(n_train), torch.ones(n_train)])[torch.randperm(2 * n_train)]) for _ in range(5)]
results["probe_accuracy_shuffled_labels"] = np.mean(shuffled, axis=0).tolist()


# ---- Difference of means and steering ------------------------------------------------------------------
def direction(layer, position):
    return acts["female_train"][0][layer, :, position].mean(0) - acts["male_train"][0][layer, :, position].mean(0)


def steered_metric(names, layer, position, vector):
    """Add a vector to the residual stream entering `layer` at one position, and read the metric."""
    def add(resid, hook):
        resid[:, position, :] += vector
        return resid
    with torch.no_grad():
        return metric(model.run_with_hooks(tokens_of(names), fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", add)]))


STEER_LAYER = 4
v = direction(STEER_LAYER, NAME_POS)
strengths = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
sweep = [steered_metric(male_test, STEER_LAYER, NAME_POS, s * v) for s in strengths]
random_sweeps = []
for _ in range(5):
    r = torch.randn_like(v); r = r / r.norm() * v.norm()
    random_sweeps.append([steered_metric(male_test, STEER_LAYER, NAME_POS, s * r) for s in strengths])
by_layer_name = [steered_metric(male_test, layer, NAME_POS, direction(layer, NAME_POS)) for layer in range(n_layers)]
by_layer_last = [steered_metric(male_test, layer, LAST, direction(layer, LAST)) for layer in range(n_layers)]

# the probe direction at the same place, rescaled to the same length
x_train = torch.cat([acts["male_train"][0][STEER_LAYER, :, NAME_POS], acts["female_train"][0][STEER_LAYER, :, NAME_POS]])
_, w = fit_probe(x_train, torch.cat([torch.zeros(n_train), torch.ones(n_train)]))
w_scaled = w / w.norm() * v.norm()
results.update({
    "steer_layer": STEER_LAYER, "direction_norm": v.norm().item(),
    "typical_stream_norm": acts["male_train"][0][STEER_LAYER, :, NAME_POS].norm(dim=-1).mean().item(),
    "strengths": strengths, "steer_difference_of_means": sweep, "steer_random_direction": np.mean(random_sweeps, axis=0).tolist(),
    "steer_by_layer_name_position": by_layer_name, "steer_by_layer_last_position": by_layer_last,
    "cosine_probe_vs_difference_of_means": torch.nn.functional.cosine_similarity(w, v, dim=0).item(),
    "steer_probe_direction_same_norm": steered_metric(male_test, STEER_LAYER, NAME_POS, w_scaled),
    "female_test_baseline": metric(acts["female_test"][1]), "male_test_baseline": metric(acts["male_test"][1]),
})
(ROOT / "code" / "results_directions.json").write_text(json.dumps(results, indent=1))

# ---------------------------------------------------------------------------------------------------
layers = np.arange(n_layers + 1)
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.plot(layers, results["probe_accuracy_name_position"], color=BLUE, lw=1.8, marker="o", ms=3.5, label="at the name")
ax.plot(layers, results["probe_accuracy_last_position"], color=RED, lw=1.8, marker="o", ms=3.5, label="at the last position")
ax.plot(layers, results["probe_accuracy_shuffled_labels"], color=MUTED, lw=1.2, ls="--", label="at the name, shuffled labels")
ax.set_xlabel("number of layers applied"); ax.set_ylabel("accuracy on held-out names"); ax.set_ylim(0.3, 1.03)
ax.set_xticks(layers[::2]); ax.legend(fontsize=7.5, loc="upper left"); grid(ax)
fig.tight_layout(); fig.savefig(FIG / "fig_probe_layers.pdf"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
ax = axes[0]
ax.axhline(0, color=MUTED, lw=0.5)
ax.plot(strengths, sweep, color=BLUE, lw=1.8, marker="o", ms=4, label="difference of means")
ax.plot(strengths, results["steer_random_direction"], color=MUTED, lw=1.4, ls="--", marker="o", ms=3, label="random direction, same length")
ax.set_xlabel("strength $\\eta$ (layer %d, at the name)" % STEER_LAYER); ax.set_ylabel("logit(she) $-$ logit(he)")
ax.legend(fontsize=7.5, loc="upper left"); grid(ax)
ax = axes[1]
ax.axhline(0, color=MUTED, lw=0.5)
ax.plot(range(n_layers), by_layer_name, color=BLUE, lw=1.8, marker="o", ms=3.5, label="steer at the name")
ax.plot(range(n_layers), by_layer_last, color=RED, lw=1.8, marker="o", ms=3.5, label="steer at the last position")
ax.set_xlabel("layer at which the direction is added ($\\eta=1$)"); ax.set_ylabel("logit(she) $-$ logit(he)")
ax.set_xticks(range(0, n_layers, 2)); ax.legend(fontsize=7.5, loc=(0.30, 0.74)); grid(ax)
fig.tight_layout(w_pad=2.0); fig.savefig(FIG / "fig_steering.pdf"); plt.close(fig)
print(json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in results.items()}, indent=0))
