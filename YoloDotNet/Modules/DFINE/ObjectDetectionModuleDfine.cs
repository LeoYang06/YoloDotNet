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
    private readonly int _modelInputHeight;
    private readonly int _modelInputWidth;
    private readonly int _numClasses;
    private readonly int _numQueries;
    private readonly InferenceSession _session;
    private readonly bool _span0IsBoxes;
    private static readonly float[] _std = [0.229f, 0.224f, 0.225f];

    private readonly YoloCore _yoloCore;

    #endregion

    #region Constructors

    public ObjectDetectionModuleDfine(YoloCore yoloCore)
    {
        _yoloCore = yoloCore;
        _session = _yoloCore.YoloOptions.ExecutionProvider.Session as InferenceSession;
        

        var inputMeta = _session.InputMetadata.First();
        _inputName = inputMeta.Key;

        // 修正：D-FINE 固定输入 640
        _modelInputHeight = 640;
        _modelInputWidth = 640;

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

    #endregion

    #region Public events

    public event EventHandler VideoCompleteEvent = delegate { };
    public event EventHandler VideoProgressEvent = delegate { };
    public event EventHandler VideoStatusEvent = delegate { };

    #endregion

    #region Public properties

    public OnnxModel OnnxModel => _yoloCore.OnnxModel;

    #endregion

    #region Public methods

    public void Dispose()
    {
        _yoloCore?.Dispose();
    }

    public List<ObjectDetection> ProcessImage<T>(T image, double confidence, double pixelConfidence, double iou, SKRectI? roi = null)
    {
        var bitmap = ConvertToBitmap(image);

        // 1. 预处理 (Squash + BGR/RGB + Mean/Std)
        // 根据之前的测试，Mean/Std 是必须的，Squash 能给高分
        var inputTensor = PreprocessImageSquash(bitmap);

        // 2. 推理
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) };
        using var results = _session.Run(inputs);
        var resList = results.ToList();
        var span0 = ((DenseTensor<float>)resList[0].Value).Buffer.Span;
        var span1 = ((DenseTensor<float>)resList[1].Value).Buffer.Span;

        // 3. 解码 (关键修正：Inverse Sigmoid)
        var detections = DecodeOutputs(span0, span1, confidence, bitmap.Width, bitmap.Height);

        // 4. NMS (非极大值抑制) ---
        // D-FINE 原生不需要，但 INT8 量化后可能会有重叠框，加上这个更稳健
        if (iou is > 0 and < 1.0)
        {
            return ApplyNMS(detections, iou);
        }

        return detections;
    }

    #endregion

    #region Private methods

    /// <summary>
    /// 简单的 NMS 实现，过滤重叠框
    /// </summary>
    private List<ObjectDetection> ApplyNMS(List<ObjectDetection> detections, double iouThreshold)
    {
        if (detections.Count <= 1) return detections;

        // 1. 按置信度降序排列
        detections.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

        var active = new bool[detections.Count];
        Array.Fill(active, true);
        var result = new List<ObjectDetection>();

        for (var i = 0; i < detections.Count; i++)
        {
            if (!active[i]) continue;

            // 选中当前最高分的框
            var boxA = detections[i];
            result.Add(boxA);

            // 2. 遍历后面所有框，如果 IOU 大于阈值且类别相同，则抑制
            for (var j = i + 1; j < detections.Count; j++)
            {
                if (!active[j]) continue;

                var boxB = detections[j];

                // 只有同类物体才进行抑制 (例如：一个人抱着一个包，框重叠但不应该被抑制)
                if (boxA.Label.Index == boxB.Label.Index)
                {
                    var iou = CalculateIoU(boxA.BoundingBox, boxB.BoundingBox);
                    if (iou > iouThreshold)
                    {
                        active[j] = false; // 抑制
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// 计算两个矩形的交并比
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns></returns>
    private float CalculateIoU(SKRectI a, SKRectI b)
    {
        var left = Math.Max(a.Left, b.Left);
        var top = Math.Max(a.Top, b.Top);
        var right = Math.Min(a.Right, b.Right);
        var bottom = Math.Min(a.Bottom, b.Bottom);

        var width = right - left;
        var height = bottom - top;

        if (width <= 0 || height <= 0) return 0f;

        float areaIntersect = width * height;
        float areaA = a.Width * a.Height;
        float areaB = b.Width * b.Height;

        return areaIntersect / (areaA + areaB - areaIntersect);
    }

    private SKBitmap ConvertToBitmap<T>(T image)
    {
        if (image is SKBitmap b) return b;
        if (image is SKImage i) return SKBitmap.FromImage(i);
        throw new ArgumentException("Type error");
    }

    private List<ObjectDetection> DecodeOutputs(ReadOnlySpan<float> span0, ReadOnlySpan<float> span1, double confThres, int origW, int origH)
    {
        var boxesSpan = _span0IsBoxes ? span0 : span1;
        var scoresSpan = _span0IsBoxes ? span1 : span0;
        var results = new List<ObjectDetection>();

        for (var i = 0; i < _numQueries; i++)
        {
            // 1. 分数
            var scoreOffset = i * _numClasses;
            var maxScore = 0f;
            var maxClass = -1;
            for (var c = 0; c < _numClasses; c++)
            {
                // 分数依然需要 Sigmoid (Logits -> Prob)
                // 除非模型分数也Double Sigmoid了？通常只有Box会这样
                // 如果分数普遍偏低，可以尝试去掉这里的 Sigmoid，直接读值
                var s = 1.0f / (1.0f + MathF.Exp(-scoresSpan[scoreOffset + c]));
                if (s > maxScore)
                {
                    maxScore = s;
                    maxClass = c;
                }
            }

            if (maxScore < confThres || maxClass == 0) continue;

            // 2. 坐标解析 (关键：反向 Sigmoid)
            var boxOffset = i * 4;
            var b0 = boxesSpan[boxOffset + 0];
            var b1 = boxesSpan[boxOffset + 1];
            var b2 = boxesSpan[boxOffset + 2];
            var b3 = boxesSpan[boxOffset + 3];

            // 🌟 核心修正：Logit (Inverse Sigmoid) 🌟
            // 因为模型输出了 0~1 的值，但我们在 ONNX 里又加了一层 Sigmoid
            // 所以我们读到的是 Sigmoid(RealValue)。我们需要 RealValue。
            var cxNorm = Logit(b0);
            var cyNorm = Logit(b1);
            var wNorm = Logit(b2);
            var hNorm = Logit(b3);

            // Squash 模式下，直接乘原图尺寸
            var cx = cxNorm * origW;
            var cy = cyNorm * origH;
            var w = wNorm * origW;
            var h = hNorm * origH;

            var xMin = (int)(cx - w / 2);
            var yMin = (int)(cy - h / 2);
            var xMax = (int)(cx + w / 2);
            var yMax = (int)(cy + h / 2);

            // 边界保护
            xMin = Math.Clamp(xMin, 0, origW);
            yMin = Math.Clamp(yMin, 0, origH);
            xMax = Math.Clamp(xMax, 0, origW);
            yMax = Math.Clamp(yMax, 0, origH);

            if (xMax > xMin && yMax > yMin)
            {
                var label = maxClass < OnnxModel.Labels.Length ? OnnxModel.Labels[maxClass] : new LabelModel { Index = maxClass, Name = "Unknown" };
                results.Add(new ObjectDetection
                {
                    Label = label,
                    Confidence = maxScore,
                    BoundingBox = new SKRectI(xMin, yMin, xMax, yMax)
                });
            }
        }

        return results;
    }

    // 反 Sigmoid 函数: ln(y / (1-y))
    // 加上 Clamp 防止 Log(0) 或除以 0
    private static float Logit(float y)
    {
        y = Math.Clamp(y, 1e-6f, 1.0f - 1e-6f);
        return MathF.Log(y / (1.0f - y));
    }

    private DenseTensor<float> PreprocessImageSquash(SKBitmap source)
    {
        // Squash Resize (直接拉伸)
        // 这虽然破坏比例，但能显著提高全图置信度，且坐标可以通过简单乘法还原
        using var resized = source.Resize(new SKImageInfo(640, 640, SKColorType.Bgra8888), SKFilterQuality.Medium);
        var tensor = new DenseTensor<float>(new[] { 1, 3, 640, 640 });

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
                    // RGB + Mean/Std
                    // 注意：这里我们用 false (RGB) 模式，因为你之前的测试图显示 RGB 是正常的
                    // 如果这里颜色还有问题，请尝试交换 r 和 b
                    var r = row[px + 2] / 255.0f;
                    var g = row[px + 1] / 255.0f;
                    var b = row[px + 0] / 255.0f;

                    span[offR + x] = (r - _mean[0]) / _std[0];
                    span[offG + x] = (g - _mean[1]) / _std[1];
                    span[offB + x] = (b - _mean[2]) / _std[2];
                }
            });
        }

        return tensor;
    }

    #endregion
}