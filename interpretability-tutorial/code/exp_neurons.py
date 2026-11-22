"""Session 4: what do single MLP neurons of GPT-2 small respond to? (About one minute on a CPU.)

Records the neurons of one MLP layer on 300 documents and lists the contexts on which chosen neurons are
most active. Writes code/results_neurons.json.
"""
import json
from pathlib import Path

import torch
from datasets import load_dataset

from common import load_model

ROOT = Path(__file__).resolve().parent.parent
model = load_model()
LAYER, HOOK = 6, "blocks.6.mlp.hook_post"
N_DOCS, CONTEXT, BATCH, KEEP = 300, 128, 25, 10
docs = []
for row in load_dataset("NeelNanda/pile-10k", split="train"):
    ids = model.to_tokens(row["text"])[0, :CONTEXT]
    if ids.shape[0] == CONTEXT:
        docs.append(ids)
    if len(docs) == N_DOCS:
        break
tokens = torch.stack(docs)

top_values = torch.full((KEEP, 3072), -1e9); top_index = torch.zeros(KEEP, 3072, dtype=torch.long)
for start in range(0, N_DOCS, BATCH):
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens[start:start + BATCH], names_filter=HOOK)
    acts = cache[HOOK][:, 1:].flatten(0, 1)                                   # skip the start token
    values, index = acts.topk(KEEP, dim=0)
    merged_values = torch.cat([top_values, values]); merged_index = torch.cat([top_index, index + start * (CONTEXT - 1)])
    top_values, order = merged_values.topk(KEEP, dim=0)
    top_index = merged_index.gather(0, order)


def context(flat_index, width=5):
    doc, pos = divmod(int(flat_index), CONTEXT - 1)
    pos += 1
    return (model.to_string(tokens[doc, max(1, pos - width):pos]) + "[" + model.to_string(tokens[doc, pos:pos + 1]) + "]").replace("\n", " ")


doc_of = lambda i: divmod(int(i), CONTEXT - 1)[0]
describe = lambda j: {"neuron": f"{LAYER}.{j}", "max_activation": round(top_values[0, j].item(), 2),
                      "documents_among_top": len({doc_of(i) for i in top_index[:, j]}),
                      "top_contexts": [context(i) for i in top_index[:, j]]}
EXPLORE = "--explore" in __import__("sys").argv                           # print the first forty neurons instead
CHOSEN = [] if EXPLORE else [35, 0, 26]                                 # the three neurons shown in the text
results = {"layer": LAYER, "n_tokens": int(tokens.shape[0] * (CONTEXT - 1)), "chosen": [describe(j) for j in CHOSEN]}
(ROOT / "code" / "results_neurons.json").write_text(json.dumps(results, indent=1))
if not CHOSEN:                                                                 # exploration: print the first 40 neurons
    for j in range(40):
        d = describe(j)
        print(j, d["documents_among_top"], "|", " ~ ".join(d["top_contexts"][:7])[:300])
else:
    print(json.dumps(results, indent=0)[:3000])
