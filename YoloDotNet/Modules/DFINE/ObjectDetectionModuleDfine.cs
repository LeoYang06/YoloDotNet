// Copyright © 2026 Zheng Yang. All rights reserved.
// Author      : Leo Yang
// Date        : 2026/01/19
// UpdatedDate : 2026/01/23

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace YoloDotNet.Modules.DFINE;

internal class ObjectDetectionModuleDfine : IObjectDetectionModule
{
    #region Private fields

    private readonly string _inputName;

    // HF ImageNet Mean/Std
    private static readonly float[] _mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] _std = [0.229f, 0.224f, 0.225f];

    private readonly int _numClasses;
    private readonly int _numQueries;
    private readonly InferenceSession _session;
    private readonly bool _span0IsBoxes;
    private readonly YoloCore _yoloCore;

    #endregion

    public OnnxModel OnnxModel => _yoloCore.OnnxModel;

    // 接口事件实现
    public event EventHandler VideoCompleteEvent = delegate { };
    public event EventHandler VideoProgressEvent = delegate { };
    public event EventHandler VideoStatusEvent = delegate { };

    public ObjectDetectionModuleDfine(YoloCore yoloCore)
    {
        _yoloCore = yoloCore;
        _session = _yoloCore.YoloOptions.ExecutionProvider.Session as InferenceSession ?? throw new InvalidOperationException("Session error");

        var inputMeta = _session.InputMetadata.First();
        _inputName = inputMeta.Key;

        var output0 = _session.OutputMetadata.ElementAt(0).Value.Dimensions;
        var output1 = _session.OutputMetadata.ElementAt(1).Value.Dimensions;

        if (output0.Last() == 4)
        {
            _span0IsBoxes = true;
            _numQueries = output0[1];
            _numClasses = output1[2];
        }
        else
        {
            _span0IsBoxes = false;
            _numQueries = output1[1];
            _numClasses = output0[2];
        }
    }

    public void Dispose()
    {
        _yoloCore?.Dispose();
    }

    public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou)
    {
        var bitmap = ConvertToBitmap(image);

        // 1. 自定义高性能预处理 (Squash + Mean/Std)
        var inputTensor = PreprocessImageSquash(bitmap);

        // 2. 推理
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) };
        using var results = _session.Run(inputs);

        var resultList = results.ToList();
        var span0 = ((DenseTensor<float>)resultList[0].Value).Buffer.Span;
        var span1 = ((DenseTensor<float>)resultList[1].Value).Buffer.Span;

        // 3. 解码 (得到原始框)
        // 租用一个临时数组来存放解码结果，避免 List 的 GC
        var tempBuffer = ArrayPool<ObjectResult>.Shared.Rent(_numQueries);
        var count = DecodeOutputsToSpan(span0, span1, confidence, bitmap.Width, bitmap.Height, tempBuffer);
        var rawSpan = tempBuffer.AsSpan(0, count);

        // 4. NMS (使用 YoloCore 高性能实现)
        // 只有当 IOU 有效时才执行 NMS
        var finalSpan = iou is > 0 and < 1.0 ? _yoloCore.RemoveOverlappingBoxes(rawSpan, iou) : rawSpan;

        // 5. 转换结果
        var detections = new List<ObjectDetection>(finalSpan.Length);
        foreach (var or in finalSpan)
        {
            detections.Add((ObjectDetection)or);
        }

        // 归还内存
        ArrayPool<ObjectResult>.Shared.Return(tempBuffer, true); // 此时 ObjectResult 引用已解绑，安全归还

        return detections;
    }

    /// <summary>
    /// 改为填充 Span，零 GC 优化
    /// </summary>
    /// <param name="span0"></param>
    /// <param name="span1"></param>
    /// <param name="confThres"></param>
    /// <param name="origW"></param>
    /// <param name="origH"></param>
    /// <param name="buffer"></param>
    /// <returns></returns>
    private int DecodeOutputsToSpan(ReadOnlySpan<float> span0, ReadOnlySpan<float> span1, double confThres, int origW, int origH, ObjectResult[] buffer)
    {
        var boxesSpan = _span0IsBoxes ? span0 : span1;
        var scoresSpan = _span0IsBoxes ? span1 : span0;
        var validCount = 0;

        for (var i = 0; i < _numQueries; i++)
        {
            var scoreOffset = i * _numClasses;
            var maxScore = 0f;
            var maxClass = -1;

            // 寻找最大分类
            for (var c = 0; c < _numClasses; c++)
            {
                // Sigmoid
                var val = scoresSpan[scoreOffset + c];
                var s = 1.0f / (1.0f + MathF.Exp(-val));
                if (s > maxScore)
                {
                    maxScore = s;
                    maxClass = c;
                }
            }

            if (maxScore < confThres || maxClass == 0) continue;

            // 坐标解析 (Inverse Sigmoid / Logit)
            var boxOffset = i * 4;
            var cx = Logit(boxesSpan[boxOffset + 0]) * origW;
            var cy = Logit(boxesSpan[boxOffset + 1]) * origH;
            var w = Logit(boxesSpan[boxOffset + 2]) * origW;
            var h = Logit(boxesSpan[boxOffset + 3]) * origH;

            var xMin = (int)(cx - w / 2);
            var yMin = (int)(cy - h / 2);
            var xMax = (int)(cx + w / 2);
            var yMax = (int)(cy + h / 2);

            xMin = Math.Clamp(xMin, 0, origW);
            yMin = Math.Clamp(yMin, 0, origH);
            xMax = Math.Clamp(xMax, 0, origW);
            yMax = Math.Clamp(yMax, 0, origH);

            if (xMax > xMin && yMax > yMin)
            {
                var label = maxClass < OnnxModel.Labels.Length ? OnnxModel.Labels[maxClass] : new LabelModel { Index = maxClass, Name = "Unknown" };

                // 填充到 Buffer
                buffer[validCount++] = new ObjectResult
                {
                    Label = label,
                    Confidence = maxScore,
                    BoundingBox = new SKRectI(xMin, yMin, xMax, yMax)
                };
            }
        }

        return validCount;
    }

    /// <summary>
    /// 反 Sigmoid 函数: ln(y / (1-y)),加上 Clamp 防止 Log(0) 或除以 0
    /// </summary>
    /// <param name="y"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Logit(float y)
    {
        y = Math.Clamp(y, 1e-6f, 1.0f - 1e-6f);
        return MathF.Log(y / (1.0f - y));
    }

    private DenseTensor<float> PreprocessImageSquash(SKBitmap source)
    {
        // Squash Resize (640x640)
        using var resized = source.Resize(new SKImageInfo(640, 640, SKColorType.Bgra8888), SKFilterQuality.Medium);
        var tensor = new DenseTensor<float>([1, 3, 640, 640]);

        unsafe
        {
            var ptr = (byte*)resized.GetPixels().ToPointer();
            Parallel.For(0, 640, y =>
            {
                var row = ptr + y * resized.RowBytes;
                var span = tensor.Buffer.Span;
                var offR = y * 640;
                var offG = offR + 640 * 640;
                var offB = offG + 640 * 640;

                for (var x = 0; x < 640; x++)
                {
                    var px = x * 4;
                    // Mean/Std + RGB
                    // 注意：测试图显示 RGB 是正常的故使用 false (RGB) 模式，如果这里颜色还有问题，尝试交换 r 和 b
                    span[offR + x] = (row[px + 2] / 255.0f - _mean[0]) / _std[0];
                    span[offG + x] = (row[px + 1] / 255.0f - _mean[1]) / _std[1];
                    span[offB + x] = (row[px + 0] / 255.0f - _mean[2]) / _std[2];
                }
            });
        }

        return tensor;
    }

    private SKBitmap ConvertToBitmap<T>(T image)
    {
        if (image is SKBitmap b) return b;
        if (image is SKImage i) return SKBitmap.FromImage(i);
        throw new ArgumentException("Type error");
    }
}