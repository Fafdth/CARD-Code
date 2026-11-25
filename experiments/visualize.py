"""论文图 3 风格：参考图、同一 LR 输入、TransFG/CARD 响应图对照。

论文未指定响应图生成算法；Grad-CAM/attention rollout 是本实现明确的选择，
不能把这段代码讲成已经找回论文原始制图方法，或已经验证其定性结论。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .common import choose_device, load_classifier, prepare_views, spec_dict, write_json


def normalize_maps(maps):
    if not torch.isfinite(maps).all():
        raise ValueError("Non-finite response map.")
    minimum = maps.flatten(1).min(dim=1).values[:, None, None]
    maximum = maps.flatten(1).max(dim=1).values[:, None, None]
    return (maps - minimum) / (maximum - minimum).clamp_min(1e-12)


def patch_maps(values, output_size):
    side = math.isqrt(values.shape[-1])
    if side * side != values.shape[-1]:
        raise ValueError("Response tokens do not form a square patch grid.")
    maps = F.interpolate(values.reshape(-1, 1, side, side), size=output_size,
                         mode="bilinear", align_corners=False).squeeze(1)
    return normalize_maps(maps)


def gradcam(classifier, inputs, targets=None):
    """【论文图 3 的实现选择】在 PSM 前最后一个完整 patch 块上计算类别相关 Grad-CAM。

    PSM 后空间网格已被部位选择打散，因此不在最终部位 token 上直接画二维图。
    用目标类别分数对特征求梯度；不执行优化器更新，也不写入参数的 .grad。
    """
    activations = []
    layer = classifier.transformer.encoder.layer[-1]
    handle = layer.register_forward_hook(lambda module, args, output: activations.append(output[0]))
    was_training = classifier.training
    classifier.eval()
    try:
        with torch.enable_grad():
            logits = classifier(inputs.detach().requires_grad_(True))
            target = logits.argmax(-1) if targets is None else torch.as_tensor(targets, device=logits.device).reshape(-1)
            if target.numel() != logits.shape[0] or (target < 0).any() or (target >= logits.shape[1]).any():
                raise ValueError("One valid zero-based class index is required per image.")
            score = logits.gather(1, target.long()[:, None]).sum()
            gradients = torch.autograd.grad(score, activations[0])[0][:, 1:]  # 排除 CLS。
            # 对 patch 位置平均梯度，得到各通道权重，再加权求和、ReLU 和上采样。
            channels = gradients.mean(dim=1, keepdim=True)
            values = (channels * activations[0][:, 1:]).sum(dim=-1).relu()
            maps = patch_maps(values.detach(), inputs.shape[-2:])
        return maps.detach(), logits.detach()
    finally:
        handle.remove()
        classifier.train(was_training)


@torch.no_grad()
def attention_rollout(classifier, inputs):
    """类别无关的另一种响应图：head 平均 + 残差单位阵 + 行归一化 + 层间连乘。"""
    attentions = []
    handles = [layer.attn.register_forward_hook(
        lambda module, args, output: attentions.append(output[1].detach()))
        for layer in classifier.transformer.encoder.layer]
    was_training = classifier.training
    classifier.eval()
    try:
        logits = classifier(inputs)
        identity = torch.eye(attentions[0].shape[-1], device=inputs.device)[None]
        joint = identity.expand(inputs.shape[0], -1, -1)
        for weights in attentions:
            attention = weights.mean(dim=1) + identity
            attention = attention / attention.sum(dim=-1, keepdim=True)
            joint = attention @ joint
        # 取最终 CLS→patch 的累计响应，再还原为空间网格；不包含 PSM 后的部位层。
        return patch_maps(joint[:, 0, 1:], inputs.shape[-2:]).detach(), logits.detach()
    finally:
        for handle in handles:
            handle.remove()
        classifier.train(was_training)


def visualize_group(group, device, method):
    """先生成一次 LR 张量，再把相同输入交给两个分类器，保证比较条件一致。"""
    baseline, baseline_spec = load_classifier(group["baseline_checkpoint"], device)
    card, spec = load_classifier(group["card_checkpoint"], device)
    if baseline_spec != spec:
        raise ValueError("Comparison checkpoints must use the same dataset and input/preprocessing configuration.")
    targets = group.get("targets")
    images = group["images"]
    if not images or (targets is not None and len(targets) != len(images)):
        raise ValueError("Provide images and, optionally, one target per image.")
    if method == "rollout" and targets is not None:
        raise ValueError("Attention rollout is class-agnostic; --targets is for Grad-CAM.")
    rows, records = [], []
    for index, path in enumerate(images):
        with Image.open(path) as image:
            reference, lr, tensor = prepare_views(image, spec)
        tensor = tensor.to(device)
        target = targets[index] if targets is not None else None
        if method == "gradcam":
            # 未指定目标时，以 CARD 预测为双方共同目标；不能各挑各的类别来作比较。
            card_map, card_logits = gradcam(card, tensor, [target] if target is not None else None)
            target = int(card_logits.argmax(-1).item()) if target is None else int(target)
            baseline_map, baseline_logits = gradcam(baseline, tensor, [target])
        else:
            card_map, card_logits = attention_rollout(card, tensor)
            baseline_map, baseline_logits = attention_rollout(baseline, tensor)
        rows.append(dict(reference=reference, lr=lr, baseline_map=baseline_map[0].cpu().numpy(),
                         card_map=card_map[0].cpu().numpy(),
                         label=f"{group.get('label', spec.dataset)} / {Path(path).name}", target=target))
        records.append(dict(image=str(path), classifier=spec_dict(spec), target_class=target,
            baseline_checkpoint=str(group["baseline_checkpoint"]), card_checkpoint=str(group["card_checkpoint"]),
            baseline_prediction=int(baseline_logits.argmax(-1).item()),
            card_prediction=int(card_logits.argmax(-1).item()),
            baseline_zero_map=bool(baseline_map.max().item() == 0), card_zero_map=bool(card_map.max().item() == 0)))
    return rows, records


def render_comparison(rows, output, method, alpha=0.45, cmap="jet", dpi=300):
    """四列输出参考/LR/基线/CARD；每张响应图独立归一化，只比较空间分布。"""
    if not rows or not 0 <= alpha <= 1 or dpi < 1:
        raise ValueError("Invalid figure rows, overlay alpha, or DPI.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)), squeeze=False)
    try:
        for index, row in enumerate(rows):
            axes[index, 0].imshow(row["reference"])
            axes[index, 1].imshow(row["lr"])
            for column, key in [(2, "baseline_map"), (3, "card_map")]:
                axes[index, column].imshow(row["lr"])
                axes[index, column].imshow(row[key], cmap=cmap, vmin=0, vmax=1, alpha=alpha)
            label = row["label"] + (f" / target {row['target']}" if row["target"] is not None else "")
            axes[index, 0].set_ylabel(label, fontsize=8)
            for axis in axes[index]:
                axis.set_xticks([])
                axis.set_yticks([])
        for axis, title in zip(axes[0], ["Reference", "Paired LR", f"TransFG / {method}", f"CARD / {method}"]):
            axis.set_title(title)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--card-checkpoint")
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--targets", type=int, nargs="+", help="Same zero-based target for both models per image; default CARD prediction.")
    parser.add_argument("--manifest", help="JSON groups for multiple datasets/checkpoint pairs; paths relative to this file.")
    parser.add_argument("--method", choices=["gradcam", "rollout"], default="gradcam")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--output", default="output/experiments/response_maps.png")
    args = parser.parse_args(argv)
    if args.manifest:
        if any([args.baseline_checkpoint, args.card_checkpoint, args.images, args.targets]):
            parser.error("Use a manifest or direct inputs, not both.")
        path = Path(args.manifest).resolve()
        groups = json.loads(path.read_text())["groups"]
        for group in groups:
            for key in ["baseline_checkpoint", "card_checkpoint"]:
                group[key] = str(path.parent / group[key])
            group["images"] = [str(path.parent / name) for name in group["images"]]
    else:
        if not args.baseline_checkpoint or not args.card_checkpoint or not args.images:
            parser.error("Provide both checkpoints and images, or use --manifest.")
        groups = [dict(baseline_checkpoint=args.baseline_checkpoint, card_checkpoint=args.card_checkpoint,
                       images=args.images, targets=args.targets)]
    if not groups:
        parser.error("At least one comparison group is required.")
    rows, records = [], []
    for group in groups:
        group_rows, group_records = visualize_group(group, choose_device(args.device), args.method)
        rows.extend(group_rows)
        records.extend(group_records)
    render_comparison(rows, args.output, args.method, args.alpha, args.cmap, args.dpi)
    if args.save_maps:
        for index, row in enumerate(rows):
            for key in ["baseline_map", "card_map"]:
                np.save(Path(args.output).with_name(f"{Path(args.output).stem}_{index}_{key}.npy"), row[key])
    write_json(Path(args.output).with_suffix(".json"), dict(method=args.method, comparisons=records,
        normalization="per image and model, independently scaled to [0,1]; zero maps preserved",
        implementation_choice="Paper does not define response-map generation; Grad-CAM/rollout are explicit choices."))
    print(f"Saved paired response maps and provenance: {args.output}")


if __name__ == "__main__":
    main()
