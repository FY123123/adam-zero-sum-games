import torch
from transformer_lens import TransformerBridge

model = TransformerBridge.boot_transformers("gpt2")   # GPT-2 small
model.enable_compatibility_mode()                      # folds LayerNorm, as older versions did

clean   = "When Mary and John went to the store, John gave a drink to"
corrupt = "When Mary and John went to the store, Mary gave a drink to"
clean_tokens, corrupt_tokens = model.to_tokens(clean), model.to_tokens(corrupt)
mary, john = model.to_single_token(" Mary"), model.to_single_token(" John")

def logit_diff(logits):                                # logit(Mary) - logit(John) at the end
    return (logits[0, -1, mary] - logits[0, -1, john]).item()

with torch.no_grad():
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    ld_clean, ld_corrupt = logit_diff(clean_logits), logit_diff(model(corrupt_tokens))
    n_layers, n_pos = model.cfg.n_layers, clean_tokens.shape[1]
    effect = torch.zeros(n_layers, n_pos)
    for layer in range(n_layers):
        name = f"blocks.{layer}.hook_resid_pre"        # residual stream entering this layer
        for pos in range(n_pos):
            def patch(resid, hook):                    # overwrite one position (clean value)
                resid[:, pos, :] = clean_cache[hook.name][:, pos, :]
                return resid
            logits = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(name, patch)])
            effect[layer, pos] = (logit_diff(logits) - ld_corrupt) / (ld_clean - ld_corrupt)

print(f"clean {ld_clean:.2f}, corrupted {ld_corrupt:.2f}")
print(effect[:, [10, 14]])                             # columns: second "John", final "to"
