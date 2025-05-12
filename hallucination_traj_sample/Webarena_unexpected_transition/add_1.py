import os
import json
from pathlib import Path

# 设置你的目标文件夹路径
folder_path = Path(
    "/home/weichenzhang/hallucination/llm_agent_hallucination_data/verified_results/unexpected_transition/gpt-4o-mini-2024-07-18"
)  # ← 替换为你的实际路径

# 遍历该文件夹下的所有 .json 文件
for file_path in folder_path.glob("*.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 只处理包含 input_step 且为整数的情况
        if "input_step" in data and isinstance(data["input_step"], int):
            data["input_step"] += 1

            # 保存修改后的数据（覆盖原文件）
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Updated input_step in: {file_path.name}")
        else:
            print(f"Skipped (missing or non-int input_step): {file_path.name}")

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
