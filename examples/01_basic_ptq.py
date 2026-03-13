"""
示例 01: NAFNet 敏感度分析 + PTQ 量化
功能：
1. 全op敏感度分析 - 使用真正的 PTQ 流程分析每个可量化的层
2. 智能跳过 - 自动推荐 skip 比例（覆盖95%敏感度）
3. 引擎适配 - 支持 tensorrt/onnxruntime/openvino/mnn 等
4. 完整 PTQ - prepare → calibrate → convert → ONNX
5. 报表输出 - 详细的表格报告和单独的 JPG 图表

Author: jihui
Date: 2026-03-13
Desc: 优化版 - 全op分析 + 引擎配置 + 自动推荐skip + 静默打印
"""
import torch
import torch.nn as nn
import sys
import os
import argparse

# 添加项目根目录和 src 目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from autoquant import (
    ModelQuantizer, 
    get_default_qconfig, 
    NAFNet_dgf,
    SensitivityAnalyzer
)
from autoquant.onnx_export import get_qconfig_for_engine, print_engine_info, get_supported_engines


def export_to_onnx(model, dummy_input, filename="quantized_model.onnx"):
    """导出模型为 ONNX 格式"""
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )


def main():
    parser = argparse.ArgumentParser(description="AutoQuant - NAFNet 全OP敏感度分析 + PTQ")
    parser.add_argument(
        "--engine", 
        type=str, 
        default="onnxruntime", 
        help=f"推理引擎选择，支持: {', '.join(get_supported_engines())}"
    )
    parser.add_argument(
        "--list-engines", 
        action="store_true", 
        help="列出所有支持的推理引擎及其配置"
    )
    
    args = parser.parse_args()
    
    if args.list_engines:
        print("=" * 80)
        print("AutoQuant - 支持的推理引擎列表")
        print("=" * 80)
        print_engine_info()
        return
    
    # ========================================================================
    # 步骤 1: 创建 NAFNet 模型和数据
    # ========================================================================
    model = NAFNet_dgf(
        img_channel=3,
        width=8,
        middle_blk_num=2,
        enc_blk_nums=[1, 2, 2, 2],
        dec_blk_nums=[2, 2, 2, 1],
    )
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)
    
    # ========================================================================
    # 步骤 2: 全OP敏感度分析
    # ========================================================================
    print(f"\n[1/6] 开始全OP敏感度分析 (Engine: {args.engine})...")
    qconfig = get_qconfig_for_engine(args.engine)
    analyzer = SensitivityAnalyzer(model, qconfig)
    calib_data = [torch.randn(1, 3, 64, 64) for _ in range(10)]
    
    # 分析所有可量化层（全OP分析）
    sensitivity_scores = analyzer.analyze(
        dummy_input,
        calib_data=calib_data
    )
    
    # ========================================================================
    # 步骤 3: 保存完整报表和图表
    # ========================================================================
    print("[2/6] 保存敏感度分析结果...")
    output_dir = os.path.join(project_root, "asset")
    analyzer.save_results(output_dir)
    
    # ========================================================================
    # 步骤 4: 获取自动推荐跳过的层（无需手动设定！）
    # ========================================================================
    quantizable_layers, skip_layers, recommendation_info = analyzer.get_recommended_layers()
    
    # 打印自动推荐信息
    if recommendation_info:
        print(f"\n    🤖 自动推荐结果:")
        print(f"       方法: {recommendation_info.get('description', 'auto')}")
        print(f"       Skip层数: {recommendation_info.get('skip_count', 0)}")
        print(f"       Skip占比: {recommendation_info.get('skip_percent', 0):.1f}%")
        print(f"       覆盖敏感度: {recommendation_info.get('coverage', 0):.1%}")
        
        alternatives = recommendation_info.get('alternatives', {})
    
    # ========================================================================
    # 步骤 5: PTQ 量化
    # ========================================================================
    print("\n[3/6] 准备量化模型...")
    quantizer = ModelQuantizer(model, qconfig)
    prepared_model = quantizer.prepare(skip_layers=set(skip_layers))
    
    print("[4/6] 校准中...")
    calib_data = [torch.randn(1, 3, 64, 64) for _ in range(50)]
    quantizer.calibrate(calib_data)
    
    print("[5/6] 转换为量化模型...")
    quantized_model = quantizer.convert()
    
    # ========================================================================
    # 步骤 6: 验证并导出
    # ========================================================================
    print("[6/6] 验证量化结果...")
    with torch.no_grad():
        original_output = model(dummy_input)
        quantized_output = quantized_model(dummy_input)

    if isinstance(quantized_output, tuple):
        quantized_output = quantized_output[0]
        original_output = original_output[0]
    
    if quantized_output.shape != original_output.shape:
        raise ValueError(f"量化输出形状 {quantized_output.shape} 与原始输出形状 {original_output.shape} 不匹配")
    
    mse = torch.nn.functional.mse_loss(original_output, quantized_output).item()
    
    # 导出 ONNX
    onnx_path = os.path.join(project_root, f"quantized_model_{args.engine}.onnx")
    export_to_onnx(quantized_model, dummy_input, onnx_path)
    
    # ========================================================================
    # 简要总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"  推理引擎: {args.engine}")
    print(f"  分析层总数: {len(sensitivity_scores)}")
    print(f"  跳过敏感层: {len(skip_layers)} (自动推荐)")
    print(f"  输出 MSE: {mse:.6f}")
    print(f"  报表目录: {output_dir}")
    print(f"  ONNX模型: {onnx_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
