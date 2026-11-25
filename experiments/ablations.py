"""论文表 VIII–X 的消融入口：默认只写实验计划，显式 --run 才启动训练。"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import train
from .common import CLASSES, write_json

ROOT = Path(__file__).resolve().parents[1]
SUITES = {"mfe": ["M0", "M1", "M2", "M3"],
          "alignment": ["D0", "D1", "D2", "D3"],
          "guidance": ["G0", "G1", "G2", "G3", "G4", "G5"]}


def variant_flags(name):
    """把论文的 M/D/G 编号转换成 train.py 参数，具体取值集中在本函数。"""
    flags = ["--mfe_adapter", "orthogonal", "--alignment_metric", "w2",
             "--guidance_mode", "constant", "--lambda_guidance", "1"]
    if name == "M0":
        # 【论文表 VIII】M0 纯 LR；M1/M2 分别为恒等/线性适配器，M3 为正交适配器。
        flags.append("--no-use_guidance")
    elif name in {"M1", "M2"}:
        flags[1] = "identity" if name == "M1" else "linear"
    elif name in {"D0", "D1", "D2"}:
        # 【论文表 IX】仅改变对齐度量；正交变换、λ=1、无 RDC 等条件保持相同。
        flags[3] = {"D0": "mse", "D1": "cosine", "D2": "kl"}[name]
    elif name in {"G0", "G1", "G2"}:
        # 【论文表 X】G0/G1/G2 是预热后的三个固定权重，预热期间仍统一 λ=1。
        flags[7] = {"G0": "0.1", "G1": "0.5", "G2": "1"}[name]
    elif name == "G3":
        flags[5] = "linear"
        flags.extend(["--lambda_final", "0.1"])
    elif name in {"G4", "G5"}:
        # G4 冻结策略参数但继续随机采样 λ；G5 根据分类反馈学习策略。
        flags[5] = "frozen" if name == "G4" else "rdc"
    elif name not in {"M3", "D3"}:
        raise ValueError("Unknown ablation variant: " + name)
    return flags


def build_plan(args):
    """生成配置和命令，不在这里读取真实数据、加载大模型或执行训练。"""
    variants = args.variants or (sum(SUITES.values(), []) if args.suite == "all" else SUITES[args.suite])
    allowed = set(sum(SUITES.values(), [])) if args.suite == "all" else set(SUITES[args.suite])
    if len(set(variants)) != len(variants) or not set(variants).issubset(allowed):
        raise ValueError("Variants must be unique members of the selected suite.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")
    if any(name.startswith("G") for name in variants) and args.rl_warmup_steps < 1:
        raise ValueError("Guidance ablations need a nonempty shared warm-up.")
    directory = Path(args.output).resolve() / f"{args.dataset}_{args.lr_size}_{args.img_size}"
    jobs = []
    for seed in args.seeds:
        common = ["--dataset", args.dataset, "--data_root", str(Path(args.data_root).resolve()),
                  "--pretrained_dir", str(Path(args.pretrained_dir).resolve()),
                  "--mae_ckpt_path", str(Path(args.mae_ckpt_path).resolve()),
                  "--img_size", str(args.img_size), "--lr_size", str(args.lr_size),
                  "--seed", str(seed), "--num_steps", str(args.num_steps),
                  "--warmup_steps", str(args.warmup_steps), "--rl_warmup_steps", str(args.rl_warmup_steps),
                  "--train_batch_size", str(args.train_batch_size), "--num_workers", str(args.num_workers),
                  "--eval_every", str(args.eval_every)]
        if args.gradient_accumulation_steps is not None:
            common.extend(["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)])
        warmup_dir = directory / f"seed_{seed}" / "shared_warmup"
        # 【论文表 X】同一种子的 G0–G5 共享这一份完整预热状态。
        shared_checkpoint = warmup_dir / "warmup_warmup.bin"

        def job(name, variant, flags, output_dir, warmup=False):
            argv = [*common, "--name", name, "--output_dir", str(output_dir), *flags]
            resolved = train.parse_args(argv)  # Validate identical training constraints without running.
            expected = dict(name=name, seed=seed, dataset=args.dataset, img_size=args.img_size,
                            lr_size=args.lr_size, total_updates=args.num_steps,
                            alignment_metric=resolved.alignment_metric, mfe_adapter=resolved.mfe_adapter,
                            guidance_mode=resolved.guidance_mode, use_guidance=resolved.use_guidance)
            return dict(name=name, variant=variant, seed=seed, warmup=warmup,
                        command=[sys.executable, str(ROOT / "train.py"), *argv],
                        metrics=str(output_dir / f"{name}_metrics.json"), expected=expected,
                        expected_configuration={key: getattr(resolved, key) for key in [
                            "train_batch_size", "gradient_accumulation_steps", "warmup_steps", "rl_warmup_steps",
                            "learning_rate", "backbone_lr_scale", "lambda_guidance", "lambda_final",
                            "policy_lr", "lambda_step_bound", "reward_scale", "reward_floor", "reward_max", "ema_alpha"]},
                        status="planned")

        if any(name.startswith("G") for name in variants):
            # stop_after=T_w 只控制停止位置；num_steps=T 保持 15,000，避免改变调度/状态。
            jobs.append(job("warmup", "shared_warmup", [*variant_flags("G2"), "--stop_after",
                            str(args.rl_warmup_steps), "--eval_every", "0"], warmup_dir, warmup=True))
        for variant in variants:
            flags = variant_flags(variant)
            if variant.startswith("G"):
                flags.extend(["--resume", str(shared_checkpoint)])
            jobs.append(job(variant, variant, flags, directory / f"seed_{seed}" / variant))
    return dict(executed=False, dataset=args.dataset, img_size=args.img_size,
                lr_size=args.lr_size, directory=str(directory), jobs=jobs,
                note="M3, D3 and G2 share a fixed-weight reference configuration; listed separately for table correspondence.")


def collect_results(plan):
    """只汇总配置匹配且实际完成的结果；未运行不能填成 0% 或论文里的准确率。"""
    rows = []
    for job in plan["jobs"]:
        if job["warmup"]:
            continue
        row = dict(variant=job["variant"], seed=job["seed"], status="not_run", accuracy_percent=None)
        path = Path(job["metrics"])
        if job.get("status") == "failed":
            row["status"] = "failed"
        elif path.is_file():
            metrics = json.loads(path.read_text())
            if (any(metrics.get(key) != value for key, value in job["expected"].items())
                    or any(metrics.get("configuration", {}).get(key) != value
                           for key, value in job["expected_configuration"].items())):
                row["status"] = "configuration_mismatch"
            elif not metrics.get("training_complete"):
                row["status"] = "incomplete"
            elif metrics.get("completed_updates") != job["expected"]["total_updates"]:
                row["status"] = "incomplete"
            else:
                accuracy = metrics.get("accuracy")
                if accuracy is None or not 0 <= accuracy <= 1:
                    raise ValueError(f"Invalid recorded accuracy: {path}")
                row.update(status="completed", accuracy_percent=accuracy * 100)
        rows.append(row)
    summaries = []
    for variant in dict.fromkeys(row["variant"] for row in rows):
        scores = [row["accuracy_percent"] for row in rows
                  if row["variant"] == variant and row["status"] == "completed"]
        summaries.append(dict(variant=variant, runs=len(scores),
            mean_accuracy_percent=statistics.mean(scores) if scores else None,
            sample_std_percent=statistics.stdev(scores) if len(scores) > 1 else None))
    return rows, summaries


def save_results(plan):
    rows, summaries = collect_results(plan)
    directory = Path(plan["directory"])
    for filename, values in [("runs.csv", rows), ("summary.csv", summaries)]:
        if values:
            with (directory / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    write_json(directory / "results.json", dict(runs=rows, summary=summaries))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", *SUITES], default="all")
    parser.add_argument("--variants", nargs="+", choices=sum(SUITES.values(), []))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--dataset", choices=list(CLASSES), default="CUB_200_2011")
    parser.add_argument("--data_root", default="./dataset")
    parser.add_argument("--pretrained_dir", default="./ViT-B_16.npz")
    parser.add_argument("--mae_ckpt_path", default="./mae_pretrain_vit_base.pth")
    parser.add_argument("--img_size", type=int, choices=[224, 448], default=224)
    parser.add_argument("--lr_size", type=int, choices=[28, 56], default=28)
    parser.add_argument("--num_steps", type=int, default=15000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--rl_warmup_steps", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--output", default="output/experiments/ablations")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Execute sequential training jobs.")
    mode.add_argument("--summarize-only", action="store_true", help="Read existing metrics without rerunning training.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    destination = Path(plan["directory"]) / "plan.json"
    if args.summarize_only or (not args.run and destination.is_file()):
        if not destination.is_file():
            raise FileNotFoundError("No existing plan to summarize.")
        previous = json.loads(destination.read_text())
        if [(job["expected"], job["expected_configuration"]) for job in previous["jobs"]] != [
                (job["expected"], job["expected_configuration"]) for job in plan["jobs"]]:
            raise ValueError("Existing plan has different settings; use a different --output directory.")
        plan = previous
        save_results(plan)
        print(f"Summarized existing results without running training: {plan['directory']}")
        return
    write_json(destination, plan)
    if args.run:
        plan["executed"] = True
        for job in plan["jobs"]:
            job["status"] = "running"
            started = time.time_ns()
            write_json(destination, plan)
            result = subprocess.run(job["command"], cwd=ROOT, check=False)
            fresh = Path(job["metrics"]).is_file() and Path(job["metrics"]).stat().st_mtime_ns >= started
            code = result.returncode or (0 if fresh else 1)
            job["status"] = "completed" if code == 0 else "failed"
            job["returncode"] = code
            write_json(destination, plan)
            if code:
                save_results(plan)
                raise SystemExit(code)
    save_results(plan)
    print(f"Saved {'executed' if args.run else 'unexecuted'} ablation plan and results: {plan['directory']}")


if __name__ == "__main__":
    main()
