"""2025 A2 mixed_precision_accumulation and benchmarking_mixed_precision(a)."""
import json
import os
from pathlib import Path

import torch
from torch import nn
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.ln(self.relu(self.fc1(x))))


def main():
    result = {"accumulation": {}, "autocast": {}}
    for name, accumulator, increment, cast in [
        ("fp32_acc_fp32_input", torch.float32, torch.float32, False),
        ("fp16_acc_fp16_input", torch.float16, torch.float16, False),
        ("fp32_acc_fp16_input", torch.float32, torch.float16, False),
        ("fp32_acc_explicit_cast_fp16_input", torch.float32, torch.float16, True),
    ]:
        total = torch.tensor(0, dtype=accumulator)
        for _ in range(1000):
            x = torch.tensor(0.01, dtype=increment)
            total += x.float() if cast else x
        result["accumulation"][name] = dict(value=total.item(), absolute_error=abs(total.item()-10))
    for dtype in [torch.float16, torch.bfloat16]:
        torch.manual_seed(336)
        model = ToyModel().cuda()
        observed = {}
        handles = []
        for name in ["fc1", "ln", "fc2"]:
            def hook(module, inputs, output, name=name):
                observed[name] = {"input": str(inputs[0].dtype), "output": str(output.dtype)}
            handles.append(getattr(model, name).register_forward_hook(hook))
        inputs = torch.randn(8, 16, device="cuda")
        targets = torch.randint(0, 4, (8,), device="cuda")
        with torch.autocast("cuda", dtype=dtype):
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            observed["parameters"] = sorted({str(p.dtype) for p in model.parameters()})
        loss.backward()
        observed.update(logits=str(logits.dtype), loss=str(loss.dtype),
                        gradients=sorted({str(p.grad.dtype) for p in model.parameters()}))
        for handle in handles:
            handle.remove()
        result["autocast"][str(dtype)] = observed
    torch.manual_seed(336)
    with torch.device("cuda"):
        model = BasicsTransformerLM(10000,128,768,12,12,3072,10000.0)
    inputs = torch.randint(0,10000,(4,128),device="cuda")
    targets = torch.randint(0,10000,(4,128),device="cuda")
    reference = model(inputs)
    reference_loss = cross_entropy(reference, targets)
    reference_loss.backward()
    selected = model.layers[0].attn.q_proj.weight if hasattr(model.layers[0], "attn") else None
    if selected is None:
        selected = next(p for name,p in model.named_parameters() if "q_proj" in name)
    reference_gradient = selected.grad.detach().clone()
    reference = reference.detach()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        candidate = model(inputs)
    candidate_loss = cross_entropy(candidate.float(), targets)
    candidate_loss.backward()
    result["small_model_numerics"] = {
        "fp32_loss":reference_loss.item(), "bf16_loss":candidate_loss.item(),
        "logits_relative_l2":((candidate.float()-reference).norm()/reference.norm()).item(),
        "logits_max_absolute_error":(candidate.float()-reference).abs().max().item(),
        "first_q_proj_gradient_relative_l2":((selected.grad-reference_gradient).norm()/reference_gradient.norm()).item(),
        "all_bf16_gradients_finite":all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
    }
    target = Path("results/steps456")/os.environ.get('CS336_RUN_NAME','v3')/'mixed_precision_probe.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
