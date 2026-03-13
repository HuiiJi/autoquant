"""
PTQ FakeQuantize - 后训练量化专用
使用 torch.fake_quantize API，确保 ONNX 导出为 QDQ 节点

Author: jihui
Date: 2026-03-13
Desc:
    正确的 PTQ 流程：
    ┌─────────────────────────────────────────────────────────┐
    │  阶段 1: CALIBRATION（校准）                               │
    │  - observer.enabled = True                                 │
    │  - 只统计 min/max，不计算 qparams，不做 fake quant！       │
    │  - 这样确保统计的是原始数据分布，不是量化后的数据！         │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │  阶段 2: CONVERT（转换）                                  │
    │  - 调用 calculate_qparams() 计算 scale/zp                │
    │  - observer.enabled = False                                │
    └─────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────┐
    │  阶段 3: INFERENCE（推理）                                │
    │  - 用计算好的 qparams 做 fake quant                       │
    │  - observer 保持禁用                                       │
    └─────────────────────────────────────────────────────────┘
"""
import torch
import torch.nn as nn
from typing import Optional
from .base import FakeQuantizeBase
from autoquant.core import QuantDtype, QScheme
from autoquant.observer import ObserverBase


class PTQFakeQuantize(FakeQuantizeBase):
    """
    PTQFakeQuantize：后训练量化专用
    
    关键设计原则：
    1. 校准阶段（observer.enabled=True）：只统计，不量化！
    2. 推理阶段（observer.enabled=False）：用统计好的 qparams 量化
    """

    def __init__(
        self,
        observer: ObserverBase = None,
        dtype: QuantDtype = QuantDtype.QUINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        ch_axis: int = 0,
        enabled: bool = True,
    ):
        if observer is not None:
            dtype = observer.dtype
            qscheme = observer.qscheme
            quant_min = observer.quant_min
            quant_max = observer.quant_max
            ch_axis = observer.ch_axis

        super().__init__(
            observer=observer,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            ch_axis=ch_axis,
            enabled=enabled,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        ⚠️  注意两个不同的 enabled 属性：
        1. self.enabled: FakeQuantize 的总开关（禁用后完全不量化，直接返回 x）
        2. self.observer.enabled: 只控制 observer 是否统计数据
        
        关键逻辑：
        - 如果 observer.enabled=True（校准阶段）：只统计，不量化
        - 如果 observer.enabled=False（推理阶段）：用统计好的 qparams 进行 fake quant
        """
        # 第一重检查：FakeQuantize 总开关（几乎不用，保留为备用）
        if not self.enabled:
            return x

        # ==========================================
        # 阶段 1: 校准阶段 - 只统计，不量化！
        # ==========================================
        if self.observer and self.observer.enabled:
            # 只统计数据，不做任何量化！
            self.observer(x)
            # 直接返回原始 x，确保后面层统计的是原始数据！
            return x

        # ==========================================
        # 阶段 2: 推理阶段 - 使用 qparams 做 fake quant
        # ==========================================
        # 如果还没有 qparams，直接返回
        if self.scale is None or self.zero_point is None:
            return x

        qmin = self.observer.quant_min if self.observer else self.quant_min
        qmax = self.observer.quant_max if self.observer else self.quant_max
        scale = self.scale
        zero_point = self.zero_point

        if self.qscheme in [QScheme.PER_CHANNEL_AFFINE, QScheme.PER_CHANNEL_SYMMETRIC]:
            if zero_point.dtype not in [torch.int32, torch.float32]:
                zero_point = zero_point.to(torch.int32)
            return torch.fake_quantize_per_channel_affine(
                x, scale=scale, zero_point=zero_point, axis=self.ch_axis, quant_min=qmin, quant_max=qmax
            )
        else:
            return torch.fake_quantize_per_tensor_affine(
                x, scale=scale, zero_point=zero_point, quant_min=qmin, quant_max=qmax
            )
