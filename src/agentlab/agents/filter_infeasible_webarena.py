import json

# 输入文件路径
input_file = "/Users/pujiayue/Documents/research/llm agent hallucination/VAB_for_hallucination/VAB-WebArena-Lite/config_files/wa/test_webarena.json"

# 输出文件路径
output_file = "infeasible_webarena_original.json"

# 读取原始 JSON 数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取符合条件的 task
infeasible_tasks = []
for task in data:
    eval_info = task.get("eval", {})
    reference_answers = eval_info.get("reference_answers", None)
    if reference_answers == {"fuzzy_match": "N/A"}:
        infeasible_tasks.append(task)

# 保存到新的 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(infeasible_tasks, f, ensure_ascii=False, indent=2)

print(f"提取完成，共找到 {len(infeasible_tasks)} 个 infeasible task，保存到 {output_file}")
