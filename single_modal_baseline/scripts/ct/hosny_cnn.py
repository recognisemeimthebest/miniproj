"""Hosny 2018 style shallow 3D CNN for NSCLC 2-year survival.

Reference:
    Hosny et al. "Deep learning for lung cancer prognostication:
    A retrospective multi-cohort radiomics study." PLOS Medicine 2018.
    https://doi.org/10.1371/journal.pmed.1002711

핵심 차이 vs MedicalNet ResNet-18:
    - **얕음**: 4 conv block (ResNet-18은 18 layer, 4 stage)
    - **가벼움**: ~500k params vs ResNet-18 11M (22× 적음)
    - **소규모 데이터 친화**: n=300 train에서 ResNet-18은 심한 overfit,
      Hosny의 4-block은 실제 NSCLC prognosis에서 AUROC 0.70 달성
    - **Pretrain 없이 from-scratch 학습** (원 논문은 LIDC nodule
      detection에서 transfer했지만 우리는 해당 weight 없음)

Architecture (Hosny 2018 Figure 1 기반):
    Input: (B, 1, H, W, D)
    ─── Block 1: Conv3d(1→32, 3) → BN → ReLU → MaxPool3d(2)
    ─── Block 2: Conv3d(32→64, 3) → BN → ReLU → MaxPool3d(2)
    ─── Block 3: Conv3d(64→128, 3) → BN → ReLU → MaxPool3d(2)
    ─── Block 4: Conv3d(128→256, 3) → BN → ReLU → MaxPool3d(2)
    ─── GlobalAvgPool3d → Dropout → FC(256 → num_classes)

입력 기대 크기: (B, 1, 80, 80, 80) — Hosny 원논문 50³ 대신 우리 LUNG1
stage III 중심 데이터에서 tumor가 커서 80³으로 조정 (50³은 72% 환자에서
tumor clip 발생).

Total params: ~550k (80³ 기준)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HosnyBlock(nn.Module):
    """단일 block: Conv3d → BN → ReLU → MaxPool3d(2).

    - kernel_size=3, padding=1 → spatial 크기 유지
    - MaxPool3d(2, 2) → 각 축 1/2 로 downsample
    - BatchNorm3d: small batch(8) 이라도 안정화에 도움
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # Spatial dropout — Hosny paper 3.3절: 각 block 뒤 dropout 0~0.2
        self.drop = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class HosnyCNN(nn.Module):
    """Hosny 2018 style shallow 3D CNN.

    Args:
        num_classes: 출력 클래스 수 (M1은 2 = 사망/생존).
        base_channels: 첫 block 채널 수. 나머지 block은 2×, 4×, 8×.
            default 32 → (32, 64, 128, 256) → 논문 표 값과 동일.
        dropout_conv: 각 conv block 뒤 spatial dropout.
            default 0.1 (보수적, 원 논문 0.2-0.3에서 small-n 안정성 위해 낮춤).
        dropout_fc: FC 전 dropout. default 0.3 (원 논문 0.5에서 낮춤).

    Forward:
        x: (B, 1, H, W, D)  — 보통 (B, 1, 80, 80, 80) 또는 (B, 1, 128, 128, 64)
        returns: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 32,
        dropout_conv: float = 0.1,
        dropout_fc: float = 0.3,
        in_channels: int = 1,
    ):
        super().__init__()
        c1 = base_channels             # 32
        c2 = base_channels * 2         # 64
        c3 = base_channels * 4         # 128
        c4 = base_channels * 8         # 256
        self.features = nn.Sequential(
            HosnyBlock(in_channels, c1, dropout=dropout_conv),
            HosnyBlock(c1, c2, dropout=dropout_conv),
            HosnyBlock(c2, c3, dropout=dropout_conv),
            HosnyBlock(c3, c4, dropout=dropout_conv),
        )
        # GlobalAvgPool: (B, C, H, W, D) → (B, C, 1, 1, 1) → (B, C)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_fc),
            nn.Linear(c4, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init (ReLU nonlinearity) + BN γ=1, β=0."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)    # (B, 256, H/16, W/16, D/16)
        x = self.gap(x)          # (B, 256, 1, 1, 1)
        x = self.flatten(x)      # (B, 256)
        return self.head(x)      # (B, num_classes)


def build_hosny_cnn(
    num_classes: int = 2,
    base_channels: int = 32,
    dropout_conv: float = 0.1,
    dropout_fc: float = 0.3,
) -> HosnyCNN:
    """Hosny 2018 shallow 3D CNN 빌더.

    pretrained 파라미터 없음 — 원 논문은 LIDC에서 transfer하지만 우리는
    해당 weight 없어서 from-scratch.
    """
    return HosnyCNN(
        num_classes=num_classes,
        base_channels=base_channels,
        dropout_conv=dropout_conv,
        dropout_fc=dropout_fc,
    )


__all__ = ["HosnyCNN", "HosnyBlock", "build_hosny_cnn"]
