"""实验公共支持：按保存配置加载 LR 分类器，复用同一图像预处理。"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from models.configs import get_b16_config
from models.modeling import VisionTransformer
from utils.paired_views import PairedViewTransform

CLASSES = {"CUB_200_2011": 200, "car": 196, "dog": 120, "nabirds": 555, "RP_281": 281}


@dataclass(frozen=True)
class ClassifierSpec:
    dataset: str = "CUB_200_2011"
    img_size: int = 224
    lr_size: int = 56
    split: str = "non-overlap"
    slide_step: int = 12

    def __post_init__(self):
        if self.dataset not in CLASSES or self.img_size not in {224, 448} or self.lr_size not in {28, 56}:
            raise ValueError("Unsupported paper dataset/resolution.")
        if self.split not in {"non-overlap", "overlap"} or self.slide_step < 1:
            raise ValueError("Invalid patch-embedding configuration.")
        if self.dataset == "RP_281" and (self.img_size, self.lr_size) != (224, 56):
            raise ValueError("RP-281 uses 56 -> 224.")


def build_classifier(spec):
    config = get_b16_config()
    config.split, config.slide_step = spec.split, spec.slide_step
    return VisionTransformer(config, img_size=spec.img_size, num_classes=CLASSES[spec.dataset])


def choose_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu or a CUDA device.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device


def load_classifier(path, device="cpu"):
    """支持 LR-only 导出文件，或从完整训练检查点提取 lr_model.*。

    不构造 HR/MFE/RDC，保证计时/可视化都使用论文推理阶段保留的分类器。
    类别数、输入尺寸和 patch 设置按保存元数据恢复，不能凭权重随意猜测。
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model" not in checkpoint or "args" not in checkpoint:
        raise ValueError("Checkpoint needs model weights and saved args metadata.")
    metadata = checkpoint["args"]
    required = {"dataset", "img_size", "lr_size", "split", "slide_step"}
    if not required.issubset(metadata):
        raise ValueError("Checkpoint is missing classifier configuration metadata.")
    spec = ClassifierSpec(**{key: metadata[key] for key in required})
    weights = checkpoint["model"]
    if any(key.startswith("lr_model.") for key in weights):
        weights = {key.removeprefix("lr_model."): value for key, value in weights.items()
                   if key.startswith("lr_model.")}
    classifier = build_classifier(spec)
    classifier.load_state_dict(weights, strict=True)
    return classifier.to(device).eval(), spec


def prepare_views(image, spec):
    """IV-A/B 的测试视图：返回参考图、配对 LR 图及归一化 LR 张量。"""
    transform = PairedViewTransform(spec, is_test=True)
    reference = transform.augment(image.convert("RGB"))
    lr = transform.degrade(reference)
    return reference, lr, transform.normalize(lr).unsqueeze(0)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def spec_dict(spec):
    return asdict(spec)
