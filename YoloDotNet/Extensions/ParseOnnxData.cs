// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    using Microsoft.ML.OnnxRuntime;

    public static class ParseOnnxData
    {
        public static OnnxModel ParseOnnx(this InferenceSession session, OnnxMetadataOverride? metadataOverride = null)
        {
            // 1. 获取原始 Shape (包含 -1)
            var rawInputs = GetInputShapes(session);
            var outputs = GetOutputShapes(session);
            var metadata = GetMetadata(session);

            // 2. 处理动态输入 Shape (清洗数据)
            // 我们需要把 {-1, 3, -1, -1} 变成 {1, 3, 640, 640}
            var sanitizedInputs = new Dictionary<string, long[]>();

            foreach (var kvp in rawInputs)
            {
                long[] shape = kvp.Value.ToArray(); // 复制一份副本

                // [0] Batch Size: 如果是 -1，设为 1
                if (shape.Length > 0 && shape[0] == -1) shape[0] = 1;

                // [2] Height: 如果是 -1，使用强制指定值
                if (shape.Length > 2 && shape[2] == -1)
                {
                    shape[2] = metadataOverride?.ForcedInputHeight ?? 640;
                }

                // [3] Width: 如果是 -1，使用强制指定值
                if (shape.Length > 3 && shape[3] == -1)
                {
                    shape[3] = metadataOverride?.ForcedInputWidth ?? 640;
                }

                sanitizedInputs.Add(kvp.Key, shape);
            }

            // 3. 决定 Version, Type, Labels (合并逻辑)
            ModelVersion version;
            if (metadataOverride?.ModelVersion != null)
            {
                version = metadataOverride.ModelVersion.Value;
            }
            else
            {
                version = metadata.TryGetValue("description", out var value) ? GetModelVersion(value) : ModelVersion.V8;
            }

            ModelType type;
            if (metadataOverride?.ModelType != null)
            {
                type = metadataOverride.ModelType.Value;
            }
            else
            {
                type = metadata.TryGetValue("task", out var value) ? GetModelType(value) : ModelType.ObjectDetection;
            }

            LabelModel[] labels;
            if (metadataOverride?.Labels != null)
            {
                labels = metadataOverride.Labels;
            }
            else if (metadata.TryGetValue("names", out var value))
            {
                labels = MapLabelsAndColors(value);
            }
            else
            {
                labels = []; // 防止空指针
            }

            // 4. 正确计算 Buffer 大小 (千万不能是 1 !)
            // 使用清洗后的 sanitizedInputs 来计算
            int totalSize = CalculateTotalInputShapeSize(sanitizedInputs.First().Value);

            // 5. 构建模型对象
            return new OnnxModel
            {
                InputShapes = sanitizedInputs, // <--- 传入清洗后的 Shape
                OutputShapes = outputs,
                CustomMetaData = metadata,
                ModelDataType = GetModelDataType(session),
                ModelType = type,
                ModelVersion = version,
                Labels = labels,
                InputShapeSize = totalSize // <--- 传入计算出的真实大小
            };
        }

        #region Helper Methods

        /// <summary>
        /// Retrieves the input shape dimensions from an ONNX model session.
        /// Dimensions are converted from int to long to support negative dynamic axes.
        /// </summary>
        private static Dictionary<string, long[]> GetInputShapes(InferenceSession session)
        {
            var shapes = new Dictionary<string, long[]>();

            foreach (var kvp in session.InputMetadata)
            {
                shapes.Add(kvp.Key, kvp.Value.Dimensions.Select(d => (long)d).ToArray());
            }

            return shapes;
        }

        /// <summary>
        /// Retrieves the output shape dimensions from an ONNX model session.
        /// </summary>
        private static Dictionary<string, int[]> GetOutputShapes(InferenceSession session)
        {
            var shapes = new Dictionary<string, int[]>();

            foreach (var kvp in session.OutputMetadata)
            {
                shapes.Add(kvp.Key, kvp.Value.Dimensions.ToArray());
            }

            return shapes;
        }

        /// <summary>
        /// Retrieves the custom metadata key-value pairs from the ONNX model session.
        /// </summary>
        private static Dictionary<string, string> GetMetadata(InferenceSession session)
        {
            try
            {
                return new Dictionary<string, string>(session.ModelMetadata.CustomMetadataMap);
            }
            catch (Exception)
            {
                return [];
            }
        }

        /// <summary>
        /// Determines the data type used by the ONNX model's input tensor.
        /// </summary>
        private static ModelDataType GetModelDataType(InferenceSession session)
        {
            try
            {
                foreach (var kvp in session.InputMetadata)
                {
                    return kvp.Value.ElementDataType == Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float16
                        ? ModelDataType.Float16
                        : ModelDataType.Float;
                }

                return ModelDataType.Float;
            }
            catch (Exception ex)
            {
                throw new YoloDotNetModelException("Failed to retrieve model data type from ONNX model.", ex);
            }
        }

        /// <summary>
        /// Maps ONNX labels to corresponding colors for visualization.
        /// </summary>
        private static LabelModel[] MapLabelsAndColors(string onnxLabelData)
        {
            var labelsDict = new Dictionary<int, string>();

            // 尝试 1: 使用 JsonDocument 解析 (AOT 兼容，无需反射)
            try
            {
                using var doc = JsonDocument.Parse(onnxLabelData);
                foreach (var prop in doc.RootElement.EnumerateObject())
                {
                    if (int.TryParse(prop.Name, out int id))
                    {
                        labelsDict[id] = prop.Value.GetString() ?? "";
                    }
                }
            }
            catch (JsonException)
            {
                // 尝试 2: 失败则回退到 Ultralytics 的非标格式解析 (旧逻辑)
                // 格式: {0: 'person', 1: 'bicycle'}
                try
                {
                    labelsDict = onnxLabelData.Trim('{', '}')
                   .Replace("'", "")
                   .Replace("\"", "") // 增强：额外去掉双引号，防止干扰
                   .Split(", ")
                   .Select(x => x.Split(": "))
                   .ToDictionary(x => int.Parse(x[0].Trim()), // 增强：Trim 防止空格报错
                    x => x[1].Trim());
                }
                catch
                {
                    // 兜底：如果还是解析不了，抛出更明确的异常或返回空
                    throw new FormatException($"Failed to parse ONNX labels: {onnxLabelData}");
                }
            }

            // 转换为数组
            return labelsDict.OrderBy(x => x.Key)
           .Select((kvp, index) => new LabelModel
            {
                Index = kvp.Key, // 使用字典里的 Key 作为 ID
                Name = kvp.Value
            })
           .ToArray();
        }

        /// <summary>
        /// Calculates the total number of elements in a tensor based on its shape dimensions.
        /// </summary>
        private static int CalculateTotalInputShapeSize(long[] shape)
        {
            if (shape.Length == 0)
                return 0;

            int shapeSize = 1;

            foreach (var dimension in shape)
            {
                if (dimension <= 0)
                    throw new YoloDotNetException($"All shape dimensions must be positive. Found invalid value: {dimension}", nameof(shape));

                shapeSize *= (int)dimension;
            }

            return shapeSize;
        }

        private static ModelType GetModelType(string modelType) => modelType switch
        {
            "classify" => ModelType.Classification,
            "detect" => ModelType.ObjectDetection,
            "obb" => ModelType.ObbDetection,
            "pose" => ModelType.PoseEstimation,
            "segment" => ModelType.Segmentation,
            _ => throw new YoloDotNetModelException("Unsupported task")
        };

        /// <summary>
        /// Get ONNX model version
        /// </summary>
        private static ModelVersion GetModelVersion(string modelDescription) => modelDescription.ToLower() switch
        {
            // YOLOv5
            var version when version.StartsWith("ultralytics yolov5") => ModelVersion.V5U,

            // YOLOv8
            var version when version.StartsWith("ultralytics yolov8") => ModelVersion.V8,
            var version when version.StartsWith("ultralytics yoloe-v8") => ModelVersion.V8E,

            // YOLOv9
            var version when version.StartsWith("ultralytics yolov9") => ModelVersion.V9,

            // YOLOv10
            var version when version.StartsWith("ultralytics yolov10") => ModelVersion.V10,

            // YOLOv11
            var version when version.StartsWith("ultralytics yolo11") => ModelVersion.V11, // Note the missing v in Yolo11
            var version when version.StartsWith("ultralytics yoloe-11") => ModelVersion.V11E, // Note the missing v in Yoloe-11

            // YOLOv12
            var version when version.StartsWith("ultralytics yolov12") => ModelVersion.V12,

            // YOLOv26
            var version when version.StartsWith("ultralytics yolo26") => ModelVersion.V26,

            // YOLO WorldV2
            var version when version.Contains("worldv2") => ModelVersion.V11,

            // RT-DETR
            var version when version.StartsWith("ultralytics rt-detr") => ModelVersion.RTDETR,

            var version when version.Contains("dfine") => ModelVersion.DFINE, // 简单匹配

            // Fallback: if version metadata is missing, treat the model as YOLOv8.
            var version when version.StartsWith("ultralytics") && !version.Contains("yolo") => ModelVersion.V8,

            _ => throw new YoloDotNetModelException("Onnx model not supported!")
        };

        #endregion
    }
}