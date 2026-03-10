"""
测试 NAFNet 模型和 PTQ 流程
"""
import sys
import os

sys.path.insert(0, '.')

import torch
import torch.nn as nn
from autoquant import (
    create_nafnet_simple,
    ModelQuantizer,
    get_histogram_qconfig,
    compute_psnr,
    compute_ssim,
)


def test_nafnet_model():
    """测试 NAFNet 模型是否能正常工作"""
    print("=" * 70)
    print("测试 1: NAFNet 模型")
    print("=" * 70)
    
    # 创建模型
    model = create_nafnet_simple(dim=64, num_blocks=6)
    print(f"✓ 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    
    return model, device


def test_ptq_workflow(model, device):
    """测试 PTQ 流程"""
    print("\n" + "=" * 70)
    print("测试 2: PTQ 流程")
    print("=" * 70)
    
    # 获取 qconfig
    qconfig = get_histogram_qconfig(is_symmetric=False)
    print(f"✓ 获取 qconfig 成功")
    
    # 创建 quantizer
    quantizer = ModelQuantizer(model, qconfig)
    print(f"✓ 创建 quantizer 成功")
    
    # 准备模型
    prepared_model = quantizer.prepare(inplace=False)
    prepared_model.to(device)
    prepared_model.eval()
    print(f"✓ 模型准备成功")
    
    # 创建校准数据
    calib_data = []
    for _ in range(10):
        calib_data.append(torch.randn(1, 3, 256, 256).to(device))
    
    # 校准
    quantizer.calibrate(calib_data, device)
    print(f"✓ 校准完成")
    
    # 转换
    quantized_model = quantizer.convert(prepared_model, inplace=False)
    quantized_model.to(device)
    quantized_model.eval()
    print(f"✓ 转换完成")
    
    return quantized_model


def test_quantization_quality(model, quantized_model, device):
    """测试量化质量"""
    print("\n" + "=" * 70)
    print("测试 3: 量化质量")
    print("=" * 70)
    
    # 创建测试数据
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    # 原始模型输出
    with torch.no_grad():
        orig_output = model(test_input)
    
    # 量化模型输出
    with torch.no_grad():
        quant_output = quantized_model(test_input)
    
    # 计算 PSNR 和 SSIM
    psnr = compute_psnr(orig_output, quant_output).item()
    ssim = compute_ssim(orig_output, quant_output).item()
    
    print(f"✓ 量化质量评估")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    return psnr, ssim


def main():
    print("=" * 70)
    print("NAFNet PTQ 测试")
    print("=" * 70)
    
    try:
        # 测试 1: NAFNet 模型
        model, device = test_nafnet_model()
        
        # 测试 2: PTQ 流程
        quantized_model = test_ptq_workflow(model, device)
        
        # 测试 3: 量化质量
        psnr, ssim = test_quantization_quality(model, quantized_model, device)
        
        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
