"""
NAFNet 模型 PTQ 测试脚本
"""
import torch
import torch.nn as nn
from autoquant import (
    get_per_channel_qconfig,
    prepare,
    convert,
    calibrate,
    ONNXExporter
)
from autoquant.special_models import NAFNet_dgf


def create_calibration_data():
    """
    创建校准数据
    """
    # 创建10个随机样本，形状为 (batch_size, channels, height, width)
    calibration_data = []
    for _ in range(10):
        # NAFNet_dgf 默认输入通道为3，这里使用256x256的输入
        data = torch.randn(1, 3, 256, 256)
        calibration_data.append(data)
    return calibration_data


def calibration_function(model, calibration_data):
    """
    校准函数
    """
    model.eval()
    with torch.no_grad():
        for data in calibration_data:
            # NAFNet_dgf 模型返回两个输出：A_lr 和 b_lr
            A_lr, b_lr = model(data)
            # 确保模型能够正常前向传播
            assert A_lr is not None
            assert b_lr is not None


def test_nafnet_dgf_ptq():
    """
    测试 NAFNet_dgf 模型的 PTQ 流程
    """
    print("开始 NAFNet_dgf 模型 PTQ 测试...")
    
    # 1. 加载模型
    print("加载 NAFNet_dgf 模型...")
    model = NAFNet_dgf()
    model.eval()
    print("模型加载成功！")
    
    # 2. 获取量化配置
    print("获取量化配置...")
    qconfig = get_per_channel_qconfig(is_symmetric=False)
    print("量化配置获取成功！")
    
    # 3. 准备 PTQ
    print("准备 PTQ...")
    model_prepared = prepare(model, qconfig)
    print("PTQ 准备成功！")
    
    # 4. 创建校准数据
    print("创建校准数据...")
    calibration_data = create_calibration_data()
    print(f"校准数据创建成功，共 {len(calibration_data)} 个样本！")
    
    # 5. 校准
    print("开始校准...")
    calibrate(model_prepared, calibration_data)
    print("校准成功！")
    
    # 6. 转换为量化模型
    print("转换为量化模型...")
    model_quantized = convert(model_prepared)
    print("量化模型转换成功！")
    
    # 7. 测试量化模型
    print("测试量化模型...")
    test_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        A_lr_quant, b_lr_quant = model_quantized(test_input)
    print(f"量化模型输出形状: A_lr={A_lr_quant.shape}, b_lr={b_lr_quant.shape}")
    print("量化模型测试成功！")
    
    # 8. 导出 ONNX
    print("导出 ONNX 模型...")
    dummy_input = torch.randn(1, 3, 256, 256)
    exporter = ONNXExporter()
    onnx_path = "nafnet_dgf_ptq.onnx"
    exporter.export(model_quantized, dummy_input, onnx_path)
    print(f"ONNX 模型导出成功！路径: {onnx_path}")
    
    print("\nNAFNet_dgf 模型 PTQ 测试完成！所有环节均成功通过。")


if __name__ == "__main__":
    test_nafnet_dgf_ptq()
