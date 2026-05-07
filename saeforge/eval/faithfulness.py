"""Faithfulness KL — token-level KL divergence between a forged model and its host."""

from __future__ import annotations

from saeforge.utils.lazy import require_extra


def faithfulness_kl(
    forged_model,
    host_model,
    prompts: list[str],
    *,
    tokenizer=None,
    max_length: int = 32,
    device: str = "cpu",
) -> float:
    """Mean per-token KL(host || forged) across ``prompts``.

    ``forged_model`` is a sae-forge ``NativeModel``; ``host_model`` is the HF
    model that was projected. Both must use the same tokenizer (passed via
    ``tokenizer`` or auto-loaded from the host's config when available).
    Returns the per-token KL averaged across all prompts and positions.
    """
    torch = require_extra("torch", "torch")
    F = torch.nn.functional

    if tokenizer is None:
        transformers = require_extra("transformers", "torch")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            host_model.config._name_or_path or "gpt2"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    forged_module = forged_model.torch_module.to(device).eval()
    host_module = host_model.to(device).eval()

    with torch.no_grad():
        forged_logits = forged_module(input_ids)
        host_out = host_module(input_ids=input_ids, attention_mask=attention_mask)
        host_logits = host_out.logits if hasattr(host_out, "logits") else host_out[0]

    log_q = F.log_softmax(forged_logits, dim=-1)
    log_p = F.log_softmax(host_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    masked = kl * attention_mask
    n_tokens = attention_mask.sum().clamp(min=1)
    return float((masked.sum() / n_tokens).item())
