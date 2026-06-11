"""
gpu_check.py
────────────
Diagnoses GPU availability, VRAM, CUDA version, and Ollama GPU visibility.
Run this first to confirm the environment before launching the pipeline.

    python scripts/gpu_check.py
"""

import sys
import subprocess
import json

def check_torch_cuda() -> dict:
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            name   = torch.cuda.get_device_name(0)
            total  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_v = torch.version.cuda
            return {
                "available": True,
                "device":    name,
                "vram_gb":   round(total, 2),
                "cuda":      cuda_v,
                "torch":     torch.__version__,
            }
        return {"available": False, "reason": "torch.cuda.is_available() returned False"}
    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}


def check_nvidia_smi() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            line = result.stdout.strip().split(",")
            return {
                "driver_ok": True,
                "gpu_name":  line[0].strip(),
                "vram_mb":   line[1].strip(),
                "driver":    line[2].strip(),
            }
        return {"driver_ok": False, "reason": result.stderr.strip()}
    except FileNotFoundError:
        return {"driver_ok": False, "reason": "nvidia-smi not found in PATH"}


def check_ollama() -> dict:
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines  = result.stdout.strip().splitlines()
            models = [l.split()[0] for l in lines[1:] if l.strip()]
            return {"running": True, "models": models}
        return {"running": False, "reason": result.stderr.strip()}
    except FileNotFoundError:
        return {"running": False, "reason": "ollama not found in PATH"}


def check_sentence_transformers() -> dict:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        _ = m.encode(["test"], batch_size=1)
        return {"ok": True, "device": str(m.device)}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def print_report(results: dict):
    SEP = "─" * 55
    print(f"\n{'═'*55}")
    print("  PolicyPipeline — GPU Diagnostics Report")
    print(f"{'═'*55}\n")

    icons = {True: "✅", False: "❌"}

    print(f"NVIDIA Driver\n{SEP}")
    smi = results["nvidia_smi"]
    ok  = smi.get("driver_ok", False)
    print(f"  Status  : {icons[ok]} {'OK' if ok else 'FAILED'}")
    if ok:
        print(f"  GPU     : {smi['gpu_name']}")
        print(f"  VRAM    : {int(smi['vram_mb'])//1024} GB ({smi['vram_mb']} MB)")
        print(f"  Driver  : {smi['driver']}")
    else:
        print(f"  Error   : {smi.get('reason')}")

    print(f"\nPyTorch CUDA\n{SEP}")
    tc = results["torch_cuda"]
    ok = tc.get("available", False)
    print(f"  Status  : {icons[ok]} {'OK' if ok else 'FAILED'}")
    if ok:
        print(f"  Device  : {tc['device']}")
        print(f"  VRAM    : {tc['vram_gb']} GB")
        print(f"  CUDA    : {tc['cuda']}")
        print(f"  Torch   : {tc['torch']}")
    else:
        print(f"  Error   : {tc.get('reason')}")

    print(f"\nOllama\n{SEP}")
    ol = results["ollama"]
    ok = ol.get("running", False)
    print(f"  Status  : {icons[ok]} {'OK' if ok else 'FAILED'}")
    if ok:
        models = ol.get("models", [])
        print(f"  Models  : {', '.join(models) if models else 'none pulled'}")
        qwen_ok = any("qwen2.5" in m for m in models)
        print(f"  Qwen2.5 : {icons[qwen_ok]} {'found' if qwen_ok else 'NOT found — run: ollama pull qwen2.5:7b'}")
    else:
        print(f"  Error   : {ol.get('reason')}")

    print(f"\nSentence Transformers (GPU embed)\n{SEP}")
    st = results["sentence_transformers"]
    ok = st.get("ok", False)
    print(f"  Status  : {icons[ok]} {'OK' if ok else 'FAILED'}")
    if ok:
        print(f"  Device  : {st['device']}")
    else:
        print(f"  Error   : {st.get('reason')}")

    print(f"\n{'═'*55}")
    all_ok = (
        results["nvidia_smi"].get("driver_ok")
        and results["torch_cuda"].get("available")
        and results["ollama"].get("running")
        and results["sentence_transformers"].get("ok")
    )
    if all_ok:
        print("  RESULT  : ✅  All systems ready. Pipeline can run.")
    else:
        print("  RESULT  : ❌  Issues found. Fix errors above before running.")
    print(f"{'═'*55}\n")

    return all_ok


if __name__ == "__main__":
    results = {
        "nvidia_smi":           check_nvidia_smi(),
        "torch_cuda":           check_torch_cuda(),
        "ollama":               check_ollama(),
        "sentence_transformers": check_sentence_transformers(),
    }
    ok = print_report(results)
    sys.exit(0 if ok else 1)
