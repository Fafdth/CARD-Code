"""论文 IV-A/IV-B：一次增强生成参考图，再从同一视图构造配对 LR。"""
import math

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class PairedViewTransform:
    """训练随机裁剪/翻转，测试中心裁剪；成对 HR/LR 始终来自同一图像区域。"""
    def __init__(self, args, is_test=False):
        self.img_size = args.img_size
        self.lr_size = getattr(args, "lr_size", None)
        if self.lr_size is None:
            ratio = getattr(args, "lr_scale_ratio", None)
            self.lr_size = self.img_size // ratio if ratio else 56
        if not 0 < self.lr_size <= self.img_size:
            raise ValueError("Require 0 < LR size <= classifier input size.")
        # 【论文 IV-A/IV-B】输入 224/448 分别将短边缩放至 256/512；全部图像缩放为双线性。
        resize = math.ceil(self.img_size * 8 / 7)
        augment = [T.Resize(resize, interpolation=InterpolationMode.BILINEAR)]
        if is_test:
            # 测试不随机翻转、不随机裁剪，保证重复评价的输入一致。
            augment.append(T.CenterCrop(self.img_size))
        else:
            augment.extend([T.RandomCrop(self.img_size), T.RandomHorizontalFlip(p=0.5)])
        self.augment = T.Compose(augment)
        # 同一参考图先下采样到 r_LR=28/56，再上采样到分类器输入尺寸。
        self.degrade = T.Compose([
            T.Resize((self.lr_size, self.lr_size), interpolation=InterpolationMode.BILINEAR),
            T.Resize((self.img_size, self.img_size), interpolation=InterpolationMode.BILINEAR),
        ])
        if getattr(args, "dataset", "") == "car":
            # Car-S 使用专用通道统计量；其他四个数据集使用 ImageNet 统计量。
            mean, std = (0.4703, 0.4598, 0.4548), (0.2881, 0.2869, 0.2942)
        else:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.normalize = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def __call__(self, image):
        # 只执行一次随机增强；禁止两条分支独立裁剪/翻转后再称作“配对”输入。
        reference = self.augment(image.convert("RGB"))
        low_resolution = self.degrade(reference)
        return self.normalize(reference), self.normalize(low_resolution)
