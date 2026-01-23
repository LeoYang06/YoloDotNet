// Copyright © 2026 Zheng Yang. All rights reserved.
// Author      : Leo Yang
// Date        : 2026/01/19
// UpdatedDate : 2026/01/20

namespace YoloDotNet.Models;

/// <summary>
/// 用于手动指定 ONNX 模型缺失的元数据
/// </summary>
public record OnnxMetadataOverride()
{
    /// <summary>
    /// 用于手动指定 ONNX 模型缺失的元数据
    /// </summary>
    public OnnxMetadataOverride(ModelVersion ModelVersion, ModelType ModelType, LabelModel[] Labels) : this()
    {
        this.ModelVersion = ModelVersion;
        this.ModelType = ModelType;
        this.Labels = Labels;
    }

    public ModelVersion? ModelVersion { get; set; }
    public ModelType? ModelType { get; set; }
    public LabelModel[]? Labels { get; set; }

    // 如果 D-FINE 模型的输入尺寸识别有误，也可以在这里强制指定，但通常不需要
    // public int? ForcedInputWidth { get; set; } 

    // --- 新增：强制指定输入分辨率 ---
    // D-FINE 默认是 640，但如果模型是动态的，我们需要在这里告诉程序用多少
    public int ForcedInputWidth { get; set; } = 640;
    public int ForcedInputHeight { get; set; } = 640;
}