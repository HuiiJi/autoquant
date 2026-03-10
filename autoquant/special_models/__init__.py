"""
特殊模型支持模块
- Transformer/LLM专用量化策略
- NAFNet等图像修复模型
- 多模态模型支持
"""
from .transformer import (
    TransformerQuantizer,
    SmoothQuantQuantizer,
    KVCacheQuantizer,
    get_transformer_qconfig,
    get_smoothquant_qconfig,
)
from .nafnet import (
    NAFNet,
    NAFBlock,
    LayerNorm2d,
    create_nafnet_simple,
    create_nafnet_denoise,
    create_nafnet_deblur,
)

__all__ = [
    'TransformerQuantizer',
    'SmoothQuantQuantizer',
    'KVCacheQuantizer',
    'get_transformer_qconfig',
    'get_smoothquant_qconfig',
    'NAFNet',
    'NAFBlock',
    'LayerNorm2d',
    'create_nafnet_simple',
    'create_nafnet_denoise',
    'create_nafnet_deblur',
]
