"""2년 생존 분류용 3D CNN (MedicalNet ResNet-18 pretrained).

기획서 §3.4 M1 (CT-only prognosis) 모델.

- Backbone: MedicalNet 3D ResNet-18 (Chen et al. 2019, arXiv:1904.00625)
  23개 의료 3D 데이터셋 pretrain. HuggingFace `TencentMedicalNet/MedicalNet-Resnet18`
  에서 MONAI가 자동 다운로드 (~125MB, .cache/huggingface).
- Head: GAP → Dropout → FC(512→num_classes)

입력: (B, 1, 128, 128, 64) float32 ∈ [0,1] (ct_preprocess.py 출력)
출력: (B, num_classes) logits

사용:
    from src.models.cnn3d import build_nsclc_classifier
    model = build_nsclc_classifier(pretrained=True)
    logits = model(image)           # (B, 2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import resnet18


class NSCLCResNet18Classifier(nn.Module):
    """MedicalNet 3D ResNet-18 + classification head."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        feature_dim: int = 512,
    ):
        super().__init__()
        # feed_forward=False → 마지막 FC 제거, avgpool 후 feature vector만 반환
        self.backbone = resnet18(
            pretrained=pretrained,
            progress=True,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,  # dummy, FC 안 씀
            feed_forward=False,
            shortcut_type="A",
            bias_downsample=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)       # (B, 512)
        return self.head(feat)         # (B, num_classes)

    def freeze_backbone(self) -> None:
        """Backbone 파라미터 고정 (head만 학습)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_nsclc_classifier(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> NSCLCResNet18Classifier:
    """기본 빌더. cfg 없이 바로 쓰기 위해."""
    return NSCLCResNet18Classifier(
        num_classes=num_classes, pretrained=pretrained, dropout=dropout
    )


__all__ = ["NSCLCResNet18Classifier", "build_nsclc_classifier"]
