"""Fixed three-condition numerical diagnostic; no fitting or metric selection."""

import argparse
import importlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.paper import residual_shift as residual
from tools.paper.benchmark_io import finish_output, reserve_output, sha256


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("probe requires isolated physical GPU 0")
    records, predictions, inputs = residual._load_locked_inputs(args.run)
    selected = residual.select_attention_rows(records)[:2]
    locked_selection = json.loads(args.failed_selection.read_text())["rows"][:2]
    if selected != locked_selection:
        raise ValueError("probe must reuse the unchanged first failed batch")
    plan = json.loads((args.run / "prepare.json").read_text())
    inputs.update(residual.execution_source_hashes(args.protocol))
    inputs.update(
        {
            str(Path(__file__).resolve()): sha256(__file__),
            str(args.failed_selection): sha256(args.failed_selection),
        }
    )
    checkpoint = Path(plan["checkpoint"])
    if sha256(checkpoint) != residual.BASE_CHECKPOINT_SHA256:
        raise ValueError("checkpoint mismatch")
    residual._assert_upstream(args.upstream)
    residual.gpu_idle()
    reserve_output(args.out)
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torchvision.io import read_image

    torch.set_num_threads(4)
    sys.path.insert(0, plan["cq_repo"])
    transform = importlib.import_module("data").eval_transform(644)
    load_model = importlib.import_module("eval").load_model
    sys.path.insert(0, str(args.upstream))
    extract_attention = importlib.import_module(
        "chessqueries.viz.attention"
    ).extract_attention
    model = load_model(checkpoint, "cuda:0").eval().requires_grad_(False)
    batch = torch.stack(
        [transform(read_image(row["image"]).float()) for row in selected]
    ).to("cuda:0")
    expected = {row["key"]: residual.fen_labels(row["pred"]) for row in predictions}
    target = np.stack([expected[row["key"]] for row in selected])
    results, arrays = [], {}
    for name, precision, backend in [
        ("high_auto", "high", None),
        ("highest_auto", "highest", None),
        ("highest_math", "highest", SDPBackend.MATH),
    ]:
        residual.gpu_idle(os.getpid())
        torch.set_float32_matmul_precision(precision)
        context = nullcontext() if backend is None else sdpa_kernel(backend)
        with context, torch.no_grad():
            before = model(batch)
            captured, maps = extract_attention(model, batch)
            after = model(batch)
        modules = [
            m
            for layer in model.decoder.layers
            for m in [layer.self_attn, layer.multihead_attn]
        ]
        grid = maps.cross_grid(layer=-1, head=None).float().cpu().numpy()
        residual.validate_attention_grid(grid, len(selected))
        result = {
            "condition": name,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "sdp_backend": "auto" if backend is None else "math",
            "dtype": str(batch.dtype),
            "capture_max_abs": float((captured - before).abs().max()),
            "restore_max_abs": float((after - before).abs().max()),
            "capture_equivalent": bool(
                torch.allclose(captured, before, atol=1e-4, rtol=1e-4)
            ),
            "restore_equivalent": bool(
                torch.allclose(after, before, atol=1e-4, rtol=1e-4)
            ),
            "hooks_restored": all("forward" not in m.__dict__ for m in modules),
            "ordinary_saved_argmax_mismatches": int(
                np.count_nonzero(before.argmax(-1).cpu().numpy() != target)
            ),
            "capture_saved_argmax_mismatches": int(
                np.count_nonzero(captured.argmax(-1).cpu().numpy() != target)
            ),
        }
        results.append(result)
        for label, tensor in [
            ("ordinary", before),
            ("capture", captured),
            ("restored", after),
        ]:
            arrays[f"{name}_{label}"] = tensor.float().cpu().numpy()
        print(json.dumps(result), flush=True)
        Path(f"{args.out}.results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )
    np.savez_compressed(f"{args.out}.logits.npz", **arrays)
    residual.verify_path_hashes(inputs, label="precision probe source/input")
    residual._assert_upstream(args.upstream)
    residual.gpu_idle(os.getpid())
    finish_output(
        args.out,
        {
            "status": "c3_precision_probe_complete",
            "release_ready": False,
            "final_test_images_read": False,
            "conditions": results,
            "inputs_sha256": inputs,
            "keys": [row["key"] for row in selected],
            "checkpoint_sha256": sha256(checkpoint),
            "gpu_used": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "upstream", "protocol", "failed-selection", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
