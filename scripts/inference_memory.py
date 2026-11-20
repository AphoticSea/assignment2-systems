"""Distinguish real no-grad inference from the training-forward benchmark."""
import json
import os
from pathlib import Path
import torch
from cs336_basics.model import BasicsTransformerLM


def main():
    folder=Path("results/steps456")/os.environ.get('CS336_RUN_NAME','v3')/'inference'
    folder.mkdir(parents=True,exist_ok=True)
    for dtype in ["fp32","bf16"]:
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.90)
        torch.manual_seed(336)
        with torch.device("cuda"):
            model=BasicsTransformerLM(10000,512,768,12,12,3072,10000.0).eval()
            inputs=torch.randint(0,10000,(4,512))
        with torch.no_grad():
            for _ in range(2):
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=dtype=="bf16"):
                    output=model(inputs)
                del output
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history(max_entries=100000)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16,enabled=dtype=="bf16"):
            output=model(inputs)
        torch.cuda.synchronize()
        result=dict(status="passed",size="small",context_length=512,batch_size=4,dtype=dtype,
                    grad_enabled=False,model_training=False,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        torch.cuda.memory._dump_snapshot(str(folder/f"small_ctx512_{dtype}.pickle"))
        torch.cuda.memory._record_memory_history(enabled=None)
        (folder/f"small_ctx512_{dtype}.json").write_text(json.dumps(result,indent=2)+"\n")
        print(result)
        del output, inputs, model


if __name__ == "__main__":
    main()
