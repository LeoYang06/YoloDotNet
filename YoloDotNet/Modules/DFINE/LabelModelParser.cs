// // Copyright © 2026-2026 Leo Yang. All rights reserved.
// // Author：leoli
// // Date：2026/01/19

using System.Text.Json.Nodes;

namespace YoloDotNet.Modules.DFINE;

public static class LabelModelParser
{
    public static LabelModel[] LoadLabelsFromConfig(string configFilePath)
    {
        var jsonString = File.ReadAllText(configFilePath);
        var jsonNode = JsonNode.Parse(jsonString);

        // 获取 id2label 节点
        var id2label = jsonNode["id2label"].AsObject();

        var labels = new List<LabelModel>();

        // 遍历 json 对象
        foreach (var kvp in id2label)
        {
            // key 是字符串 "0", "1", value 是 "None", "Person"
            int index = int.Parse(kvp.Key);
            string name = kvp.Value.ToString();

            labels.Add(new LabelModel
            {
                Index = index,
                Name = name
            });
        }

        // 确保按 Index 排序，因为数组下标必须对应模型输出的 Index
        return labels.OrderBy(x => x.Index).ToArray();
    }
}