import os
import gzip
import pickle
import glob
import json
import shutil
from pathlib import Path
from browsergym.experiments import StepInfo
from collections import defaultdict


def find_repetitive_actions(trajectories_dir, output_file=None, move_to_dir=None, map_dir=None):
    """
    查找轨迹中连续第三次及以上重复的动作

    Args:
        trajectories_dir: 包含所有任务轨迹文件夹的目录路径
        output_file: 输出结果的JSON文件路径
        move_to_dir: 移动含有连续重复动作的文件夹到此目录
        map_dir: 生成截断映射的目录路径
    """
    results = []
    # 用于统计各重复次数的计数
    repetition_counts = defaultdict(int)
    # 用于生成各重复次数的截断映射
    truncate_maps = defaultdict(dict)
    # 记录被排除的重复动作
    excluded_actions = []
    # 已移动的任务集合，避免重复移动
    moved_tasks = set()

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

        # 加载所有步骤数据
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

            except Exception as e:
                print(f"处理文件 {step_file} 时出错: {str(e)}")

        # 标记是否在此任务中找到了连续重复动作
        task_has_repetitive = False
        # 存储任务中每种重复次数的最早发生步骤
        task_repetition_steps = defaultdict(list)

        # 重新设计寻找连续重复动作的逻辑
        if len(all_steps_data) >= 3:  # 至少需要3个步骤才能有连续重复
            for i in range(len(all_steps_data) - 2):  # 遍历到倒数第三个元素
                # 检查当前步骤和后续步骤的动作是否相同
                current_action = None
                if hasattr(all_steps_data[i][1], "action"):
                    current_action = all_steps_data[i][1].action
                else:
                    continue

                # 如果当前动作为空，则跳过
                if current_action is None:
                    continue

                # 将动作转换为字符串用于比较
                current_action_str = str(current_action)

                # 统计连续相同动作的数量
                consecutive_count = 1
                consecutive_steps = [all_steps_data[i][0]]  # 记录连续步骤的编号

                # 向后查找连续相同的动作
                for j in range(i + 1, len(all_steps_data)):
                    next_action = None
                    if hasattr(all_steps_data[j][1], "action"):
                        next_action = all_steps_data[j][1].action
                    else:
                        break

                    # 如果下一步动作为空或者与当前动作不同，则中断连续计数
                    if next_action is None or str(next_action) != current_action_str:
                        break

                    consecutive_count += 1
                    consecutive_steps.append(all_steps_data[j][0])

                # 只关注第3次到第10次连续重复
                if 3 <= consecutive_count <= 10:
                    for repeat_index in range(2, consecutive_count):  # 从第3次开始（索引2）
                        current_step_num = consecutive_steps[repeat_index]
                        current_step_data = all_steps_data[i + repeat_index][1]

                        # 更新统计计数
                        repetition_counts[repeat_index + 1] += 1  # 转换为第几次重复（索引+1）
                        # 标记任务含有连续重复动作
                        task_has_repetitive = True

                        # 提取当前步骤的思考过程
                        current_thought = None
                        if (
                            hasattr(current_step_data, "agent_info")
                            and current_step_data.agent_info
                        ):
                            if hasattr(current_step_data.agent_info, "think"):
                                current_thought = current_step_data.agent_info.think
                            elif (
                                isinstance(current_step_data.agent_info, dict)
                                and "think" in current_step_data.agent_info
                            ):
                                current_thought = current_step_data.agent_info["think"]

                        # 构建结果条目
                        repetition_entry = {
                            "task_name": task_name,
                            "current_step": f"step_{current_step_num}",
                            "current_step_num": current_step_num,
                            "current_step_thought": current_thought,
                            "repeat_action": current_action,
                            "repeat_action_str": current_action_str[
                                :100
                            ],  # 添加截短的动作字符串用于展示
                            "repeat_steps": [f"step_{s}" for s in consecutive_steps[:repeat_index]],
                            "repetition_count": repeat_index + 1,  # 转换为第几次重复
                            "consecutive": True,  # 标记为连续重复
                        }

                        results.append(repetition_entry)

                        # 将当前重复情况添加到任务的对应重复次数列表
                        task_repetition_steps[repeat_index + 1].append(repetition_entry)

                        print(
                            f"找到第{repeat_index + 1}次连续重复 - 任务: {task_name}, 步骤: step_{current_step_num}, 动作: {current_action_str[:50]}..."
                        )

                # 如果找到了连续重复，则跳过已经处理过的连续步骤
                if consecutive_count > 1:
                    i += consecutive_count - 1

        # 如果该任务有连续重复动作且需要移动文件夹
        if task_has_repetitive and move_to_dir and task_name not in moved_tasks:
            target_dir = os.path.join(move_to_dir, task_name)

            # 检查目标目录是否已存在
            if not os.path.exists(target_dir):
                try:
                    # 移动任务文件夹到目标位置
                    shutil.copytree(task_dir, target_dir)
                    moved_tasks.add(task_name)
                    print(f"已复制任务 {task_name} 的文件夹到 {target_dir}")
                except Exception as e:
                    print(f"复制任务文件夹 {task_name} 时出错: {str(e)}")
            else:
                print(f"目标目录 {target_dir} 已存在，跳过复制")

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
                map_file = os.path.join(
                    map_dir, f"consecutive_repetitive_{repeat_count}_truncate_map.json"
                )
                try:
                    with open(map_file, "w", encoding="utf-8") as f:
                        json.dump(truncate_map, f, ensure_ascii=False, indent=2)
                    print(f"第{repeat_count}次连续重复的截断映射已保存到 {map_file}")
                except Exception as e:
                    print(f"保存第{repeat_count}次连续重复的截断映射时出错: {str(e)}")

    return results, repetition_counts, excluded_actions, truncate_maps


if __name__ == "__main__":
    # 设置轨迹目录路径
    trajectories_dir = (
        "/home/weichenzhang/hallucination/AL_for_hallucination/src/agentlab/agents/test_results"
    )

    # 设置输出文件路径
    output_file = os.path.join(trajectories_dir, "repetition_analysis.json")

    # 设置移动目标目录
    move_to_dir = (
        "/home/weichenzhang/hallucination/AL_for_hallucination/hallucination_traj_sample/WorkArena"
    )

    # 设置映射输出目录
    map_dir = os.path.join(trajectories_dir, "repetition_maps")

    # 确保目标目录存在
    if move_to_dir:
        os.makedirs(move_to_dir, exist_ok=True)

    # 如果目录存在就执行查找
    if os.path.exists(trajectories_dir):
        repetitions, repetition_counts, excluded_actions, truncate_maps = find_repetitive_actions(
            trajectories_dir, output_file, move_to_dir, map_dir
        )

        print("\n所有连续重复动作分析结果:")
        print("=" * 80)
        for entry in repetitions:
            print(f"任务: {entry['task_name']}")
            print(f"当前步骤: {entry['current_step']}")
            print(f"重复次数: 第{entry['repetition_count']}次")
            print(f"重复步骤列表: {entry['repeat_steps']}")
            print(f"重复动作: {entry['repeat_action_str']}...")
            print("-" * 80)

        print(f"总共找到 {len(repetitions)} 个连续重复动作")

        # 打印各重复次数的统计结果
        print("\n各连续重复次数统计:")
        print("=" * 50)
        for i in range(3, 11):
            count = repetition_counts.get(i, 0)
            print(f"第 {i} 次连续重复: {count} 个")
            if i in truncate_maps:
                print(f"生成了 {len(truncate_maps[i])} 个任务的第{i}次连续重复截断映射")
        print("=" * 50)

        # 打印被排除的重复动作
        if excluded_actions:
            print("\n被排除的连续重复动作:")
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
            print(f"总共排除了 {len(excluded_actions)} 个连续重复动作")

        print(f"详细分析结果已保存到 {output_file}")
        print(f"各连续重复次数的截断映射已保存到 {map_dir} 目录")
        if move_to_dir:
            print(f"包含连续重复动作的任务已复制到 {move_to_dir}")
    else:
        print(f"错误: 目录 {trajectories_dir} 不存在")
