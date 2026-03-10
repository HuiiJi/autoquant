"""
旷视科技 NAFNet 模型结构
论文: https://arxiv.org/abs/2204.04676
官方代码: https://github.com/megvii-research/NAFNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class LayerNorm2d(nn.LayerNorm):
    """2D LayerNorm 适配"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, 
            self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class NAFBlock(nn.Module):
    """
    NAFNet 基础模块
    包含: Simple Gate Unit + Channel Attention
    """
    def __init__(
        self,
        dim: int,
        expansion_factor: float = 2.0,
        kernel_size: int = 3,
        use_attention: bool = True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        
        self.norm1 = LayerNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0)
        
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, 
            kernel_size=kernel_size, padding=kernel_size//2,
            groups=hidden_dim
        )
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0),
            nn.Sigmoid()
        ) if use_attention else nn.Identity()
        
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.norm1(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.dwconv(x)
        x = x * self.sca(x)
        x = self.conv2(x)
        
        return identity + x * self.beta


class NAFNet(nn.Module):
    """
    完整的 NAFNet 模型结构
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 64,
        num_blocks: List[int] = [4, 6, 6, 8],
        expansion_factor: float = 2.0,
        kernel_size: int = 3,
        use_attention: bool = True,
        use_downsample: bool = True
    ):
        super().__init__()
        
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 输入卷积
        self.in_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        current_dim = dim
        
        for i, num_block in enumerate(num_blocks[:-1]):
            blocks = nn.ModuleList([
                NAFBlock(current_dim, expansion_factor, kernel_size, use_attention)
                for _ in range(num_block)
            ])
            self.encoder.append(blocks)
            
            # Downsample
            if use_downsample:
                self.encoder.append(
                    nn.Conv2d(current_dim, current_dim * 2, kernel_size=2, stride=2)
                )
                current_dim *= 2
        
        # Bottleneck
        bottleneck = nn.ModuleList([
            NAFBlock(current_dim, expansion_factor, kernel_size, use_attention)
            for _ in range(num_blocks[-1])
        ])
        self.encoder.append(bottleneck)
        
        # Decoder
        self.decoder = nn.ModuleList()
        
        for i, num_block in enumerate(reversed(num_blocks[:-1])):
            # Upsample
            if use_downsample:
                self.decoder.append(
                    nn.ConvTranspose2d(current_dim, current_dim // 2, kernel_size=2, stride=2)
                )
                current_dim = current_dim // 2
            
            blocks = nn.ModuleList([
                NAFBlock(current_dim, expansion_factor, kernel_size, use_attention)
                for _ in range(num_block)
            ])
            self.decoder.append(blocks)
        
        # 输出卷积
        self.out_conv = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 输入
        x = self.in_conv(x)
        
        # Encoder
        encoder_features = []
        
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.ModuleList):
                for block in layer:
                    x = block(x)
                encoder_features.append(x)
            else:
                x = layer(x)
        
        # Decoder
        feature_idx = len(encoder_features) - 2
        
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                # Skip connection
                if feature_idx >= 0:
                    x = x + encoder_features[feature_idx]
                    feature_idx -= 1
            else:
                for block in layer:
                    x = block(x)
        
        # 输出
        x = self.out_conv(x)
        return x + identity


def create_nafnet_simple(
    in_channels: int = 3,
    out_channels: int = 3,
    dim: int = 64,
    num_blocks: int = 6
) -> NAFNet:
    """
    创建简化的 NAFNet 模型（只有编码器，没有下采样）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        dim: 基础通道数
        num_blocks: 块数量
    
    Returns:
        NAFNet 模型
    """
    return NAFNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=[num_blocks],
        use_downsample=False
    )


def create_nafnet_denoise(
    in_channels: int = 3,
    out_channels: int = 3,
    dim: int = 64
) -> NAFNet:
    """
    创建用于图像去噪的 NAFNet 模型
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        dim: 基础通道数
    
    Returns:
        NAFNet 模型
    """
    return NAFNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=[4, 6, 6, 8],
        use_downsample=True
    )


def create_nafnet_deblur(
    in_channels: int = 3,
    out_channels: int = 3,
    dim: int = 96
) -> NAFNet:
    """
    创建用于图像去模糊的 NAFNet 模型
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        dim: 基础通道数
    
    Returns:
        NAFNet 模型
    """
    return NAFNet(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=[4, 6, 6, 8],
        use_downsample=True
    )
