#!/usr/bin/env python3
import json
import os


def find_shopping_only_tasks(file_path):
    """
    从指定的 JSON 文件中找出所有 sites 数组中只包含 "shopping" 的任务

    Args:
        file_path: JSON 文件的路径

    Returns:
        包含符合条件任务的列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []

    shopping_only_tasks = []

    for task in data:
        # 检查 sites 数组是否只包含一个元素且该元素为 "shopping"
        if task.get("sites") == ["shopping"]:
            # 添加到结果列表
            shopping_only_tasks.append(
                {
                    "task_id": task.get("task_id"),
                    "intent": task.get("intent"),
                    "sites": task.get("sites"),
                }
            )

    return shopping_only_tasks


def main():
    # 文件路径
    file_path = (
        "AL_for_hallucination/hallucination_traj_sample/webarena_popup/shopping/test.raw.json"
    )

    # 确保文件存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
        return

    # 获取所有符合条件的任务
    shopping_only_tasks = find_shopping_only_tasks(file_path)
    results = []
    for task in shopping_only_tasks:
        task_name = f"webarena.{task['task_id']}"
        results.append(task_name)

    print(results)


if __name__ == "__main__":
    main()
