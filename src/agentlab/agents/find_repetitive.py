import os
import gzip
import pickle
import glob
import json
from pathlib import Path
from browsergym.experiments import StepInfo
from collections import defaultdict


def find_repetitive_actions(trajectories_dir, output_file=None, map_dir=None):
    """
    查找轨迹中第三次及以上重复的动作

    Args:
        trajectories_dir: 包含所有任务轨迹文件夹的目录路径
        output_file: 输出结果的JSON文件路径
        map_dir: 生成截断映射的目录路径
    """
    results = []
    # 用于统计各重复次数的计数
    repetition_counts = defaultdict(int)
    # 用于生成各重复次数的截断映射
    truncate_maps = defaultdict(dict)
    # 记录被排除的重复动作
    excluded_actions = []

    # 使用glob模式查找所有任务轨迹目录
    task_dirs = []
    for item in os.listdir(trajectories_dir):
        item_path = os.path.join(trajectories_dir, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            if item in ["WorkArena", "Webarena"]:
                # 在WorkArena和Webarena子目录中找到实际任务目录
                for task_item in os.listdir(item_path):
                    task_item_path = os.path.join(item_path, task_item)
                    if os.path.isdir(task_item_path) and not task_item.startswith("."):
                        task_dirs.append(task_item_path)
            else:
                task_dirs.append(item_path)

    print(f"找到 {len(task_dirs)} 个任务轨迹目录进行分析")

    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        step_files = glob.glob(os.path.join(task_dir, "step_*.pkl.gz"))

        if not step_files:
            print(f"警告: 在 {task_dir} 中没有找到 step 文件")
            continue

        # 对step文件按编号排序
        step_files = sorted(
            step_files, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )

        # 用于跟踪每个动作的出现次数和步骤
        action_counts = defaultdict(int)
        action_steps = defaultdict(list)

        # 第一次加载所有步骤数据，构建动作历史记录
        all_steps_data = []

        for step_file in step_files:
            step_num = int(os.path.basename(step_file).split("_")[1].split(".")[0])

            try:
                with gzip.open(step_file, "rb") as f:
                    step_data = pickle.load(f)

                if not isinstance(step_data, StepInfo):
                    print(f"警告: {step_file} 中的数据不是StepInfo类型")
                    continue

                all_steps_data.append((step_num, step_data))

                # 获取动作内容
                if hasattr(step_data, "action"):
                    action = step_data.action
                    # 将动作转换为不可变类型以用作字典键
                    action_str = str(action)
                    # 记录动作出现次数和对应步骤
                    action_counts[action_str] += 1
                    action_steps[action_str].append(step_num)

            except Exception as e:
                print(f"处理文件 {step_file} 时出错: {str(e)}")

        # 存储任务中每种重复次数的最早发生步骤
        task_repetition_steps = defaultdict(list)

        # 第二次遍历，找出第三次到第十次的重复动作
        for step_num, step_data in all_steps_data:
            if hasattr(step_data, "action"):
                action = step_data.action
                action_str = str(action)

                # 获取当前动作的重复索引（当前是第几次执行这个动作）
                current_repeat_index = action_steps[action_str].index(step_num) + 1

                # 关注第3次到第10次重复
                if 3 <= current_repeat_index <= 10:
                    # 更新统计计数
                    repetition_counts[current_repeat_index] += 1

                    # 提取当前步骤的思考过程
                    current_thought = None
                    if hasattr(step_data, "agent_info") and step_data.agent_info:
                        if hasattr(step_data.agent_info, "think"):
                            current_thought = step_data.agent_info.think
                        elif (
                            isinstance(step_data.agent_info, dict)
                            and "think" in step_data.agent_info
                        ):
                            current_thought = step_data.agent_info["think"]

                    # 构建结果条目
                    repetition_entry = {
                        "task_name": task_name,
                        "current_step": f"step_{step_num}",
                        "current_step_num": step_num,  # 添加数字步骤编号，方便排序
                        "current_step_thought": current_thought,
                        "repeat_action": action,
                        "repeat_action_str": action_str[:100],  # 添加截短的动作字符串用于展示
                        "repeat_steps": [
                            f"step_{s}"
                            for s in action_steps[action_str][: current_repeat_index - 1]
                        ],
                        "repetition_count": current_repeat_index,
                    }

                    results.append(repetition_entry)

                    # 将当前重复情况添加到任务的对应重复次数列表
                    task_repetition_steps[current_repeat_index].append(repetition_entry)

                    print(
                        f"找到第{current_repeat_index}次重复 - 任务: {task_name}, 步骤: step_{step_num}, 动作: {action_str[:50]}..."
                    )

        # 处理每个重复次数的截断映射
        for repeat_count, entries in task_repetition_steps.items():
            if entries:
                # 按步骤编号排序，选择最早的那个
                entries.sort(key=lambda x: x["current_step_num"])
                earliest_entry = entries[0]

                # 添加到对应重复次数的截断映射
                truncate_maps[repeat_count][task_name] = earliest_entry["current_step_num"]

                # 记录被排除的重复动作
                if len(entries) > 1:
                    for excluded_entry in entries[1:]:
                        excluded_actions.append(
                            {
                                "task_name": task_name,
                                "repetition_count": repeat_count,
                                "selected_step": earliest_entry["current_step_num"],
                                "selected_action": earliest_entry["repeat_action_str"],
                                "excluded_step": excluded_entry["current_step_num"],
                                "excluded_action": excluded_entry["repeat_action_str"],
                            }
                        )

    # 如果提供了输出文件路径，将结果保存为JSON
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到 {output_file}")
        except Exception as e:
            print(f"保存结果到文件时出错: {str(e)}")

    # 如果提供了映射目录，生成各重复次数的截断映射
    if map_dir:
        os.makedirs(map_dir, exist_ok=True)
        for repeat_count, truncate_map in truncate_maps.items():
            if truncate_map:
                map_file = os.path.join(map_dir, f"repetitive_{repeat_count}_truncate_map.json")
                try:
                    with open(map_file, "w", encoding="utf-8") as f:
                        json.dump(truncate_map, f, ensure_ascii=False, indent=2)
                    print(f"第{repeat_count}次重复的截断映射已保存到 {map_file}")
                except Exception as e:
                    print(f"保存第{repeat_count}次重复的截断映射时出错: {str(e)}")

    return results, repetition_counts, excluded_actions, truncate_maps


if __name__ == "__main__":
    # 设置轨迹目录路径
    trajectories_dir = (
        "/home/weichenzhang/hallucination/AL_for_hallucination/src/agentlab/agents/test_results"
    )

    # 设置输出文件路径
    output_file = os.path.join(trajectories_dir, "repetition_analysis.json")

    # 设置映射输出目录
    map_dir = os.path.join(trajectories_dir, "repetition_maps")

    # 如果目录存在就执行查找
    if os.path.exists(trajectories_dir):
        repetitions, repetition_counts, excluded_actions, truncate_maps = find_repetitive_actions(
            trajectories_dir, output_file, map_dir
        )

        print("\n所有重复动作分析结果:")
        print("=" * 80)
        for entry in repetitions:
            print(f"任务: {entry['task_name']}")
            print(f"当前步骤: {entry['current_step']}")
            print(f"重复次数: 第{entry['repetition_count']}次")
            print(f"重复步骤列表: {entry['repeat_steps']}")
            print(f"重复动作: {entry['repeat_action_str']}...")
            print("-" * 80)

        print(f"总共找到 {len(repetitions)} 个重复动作")

        # 打印各重复次数的统计结果
        print("\n各重复次数统计:")
        print("=" * 50)
        for i in range(3, 11):
            count = repetition_counts.get(i, 0)
            print(f"第 {i} 次重复: {count} 个")
            if i in truncate_maps:
                print(f"生成了 {len(truncate_maps[i])} 个任务的第{i}次重复截断映射")
        print("=" * 50)

        # 打印被排除的重复动作
        if excluded_actions:
            print("\n被排除的重复动作:")
            print("=" * 80)
            for excluded in excluded_actions:
                print(
                    f"任务: {excluded['task_name']}, 重复次数: 第{excluded['repetition_count']}次"
                )
                print(
                    f"选择的步骤: step_{excluded['selected_step']}, 动作: {excluded['selected_action']}"
                )
                print(
                    f"排除的步骤: step_{excluded['excluded_step']}, 动作: {excluded['excluded_action']}"
                )
                print("-" * 80)
            print(f"总共排除了 {len(excluded_actions)} 个重复动作")

        print(f"详细分析结果已保存到 {output_file}")
        print(f"各重复次数的截断映射已保存到 {map_dir} 目录")
    else:
        print(f"错误: 目录 {trajectories_dir} 不存在")
