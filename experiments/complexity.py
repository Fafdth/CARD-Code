"""论文表 VII 的统计入口：只统计 LR 分类器，不包含训练专用的 HR/MFE/RDC。"""
import argparse
from contextlib import nullcontext
import csv
from pathlib import Path
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from .common import (ClassifierSpec, CLASSES, build_classifier, choose_device,
                     load_classifier, spec_dict, write_json)


def count_operations(model, inputs):
    """【论文表 VII】计数受支持的前向矩阵乘法/卷积，并按 batch 大小折算到每图。

    约定 1 MAC=2 FLOPs；同时报告两种口径，不强行凑论文数值。
    softmax/LayerNorm 等未计数操作不声称已覆盖；算子明细随 JSON 保存。
    """
    was_training = model.training
    model.eval()
    try:
        # PyTorch 2.5 的 inference_mode 会绕过此计数器；这里只用 no_grad 做计数。
        with torch.inference_mode(False), torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(inputs)
        total = counter.get_total_flops()
        operators = {str(op): int(value) for op, value in
                     counter.get_flop_counts().get("Global", {}).items()}
        if total <= 0:
            raise RuntimeError("No supported FLOPs were counted; do not report this as zero complexity.")
    finally:
        model.train(was_training)
    batch = inputs.shape[0]
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                flops_per_image=total / batch, gflops_per_image=total / batch / 1e9,
                macs_per_image=total / batch / 2, gmacs_per_image=total / batch / 2e9,
                operator_flops_per_batch=operators,
                convention="1 multiply-accumulate = 2 FLOPs; includes supported attention/PSM matmuls; "
                           "excludes uncounted pointwise/normalization/softmax operators")


def measure_latency(model, inputs, warmup=50, iterations=200):
    """【论文表 VII】预热后计纯前向耗时；输入已在设备上，不计读图/预处理/传输。

    CUDA 用事件并同步，CPU 用 perf_counter；批次耗时除以 B 才是 ms/image。
    这是批次平均每图耗时，不等同于单张请求（B=1）的延迟。
    """
    if warmup < 0 or iterations < 1 or inputs.shape[0] < 1:
        raise ValueError("Require warmup>=0, iterations>=1 and a nonempty batch.")
    device = inputs.device
    was_training = model.training
    model.eval()
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    try:
        with context, torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(warmup):
                model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                events = []
                for _ in range(iterations):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(inputs)
                    end.record()
                    events.append((start, end))
                torch.cuda.synchronize(device)
                times = [start.elapsed_time(end) for start, end in events]
                peak = torch.cuda.max_memory_allocated(device)
            else:
                times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    model(inputs)
                    times.append((time.perf_counter() - start) * 1000)
                peak = None
    finally:
        model.train(was_training)
    mean = statistics.mean(times)
    return dict(warmup=warmup, iterations=iterations, batch_size=inputs.shape[0],
                mean_batch_ms=mean, median_batch_ms=statistics.median(times),
                std_batch_ms=statistics.pstdev(times),
                mean_ms_per_image=mean / inputs.shape[0],
                throughput_images_per_second=1000 * inputs.shape[0] / mean,
                cuda_peak_allocated_bytes=peak,
                scope="forward only; normalized input already on device; excludes loading/transfers")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="LR export or full CARD checkpoint; omitted for architecture-only profiling.")
    parser.add_argument("--dataset", choices=list(CLASSES), default=None)
    parser.add_argument("--img_size", type=int, choices=[224, 448], default=None)
    parser.add_argument("--lr_size", type=int, choices=[28, 56], default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--flops-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/experiments/complexity.json")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("Invalid batch size or iteration count.")
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    if args.checkpoint:
        model, spec = load_classifier(args.checkpoint, device)
        for field in ["dataset", "img_size", "lr_size"]:
            value = getattr(args, field)
            if value is not None and value != getattr(spec, field):
                parser.error(f"{field} conflicts with the checkpoint metadata.")
    else:
        spec = ClassifierSpec(args.dataset or "CUB_200_2011", args.img_size or 224, args.lr_size or 56)
        model = build_classifier(spec).to(device).eval()
    inputs = torch.randn(args.batch_size, 3, spec.img_size, spec.img_size, device=device)
    operations = count_operations(model, inputs)
    latency = None if args.flops_only else measure_latency(model, inputs, args.warmup, args.iterations)
    result = dict(classifier=spec_dict(spec), checkpoint=args.checkpoint, weights_loaded=bool(args.checkpoint),
                  scope="LR classifier only; no HR encoder, MFE or RDC",
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  torch_version=str(torch.__version__), cpu_threads=torch.get_num_threads(),
                  dtype=str(inputs.dtype), cuda_version=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  operations=operations, latency=latency)
    write_json(args.output, result)
    summary = dict(parameters=operations["parameters"], gmacs_per_image=operations["gmacs_per_image"],
                   gflops_per_image=operations["gflops_per_image"], device=str(device),
                   mean_ms_per_image=latency["mean_ms_per_image"] if latency else None)
    with Path(args.output).with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(f"Saved LR-only profiling metadata: {args.output}")


if __name__ == "__main__":
    main()
