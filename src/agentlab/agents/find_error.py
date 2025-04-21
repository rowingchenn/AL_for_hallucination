import os
import gzip
import pickle
import glob
import json
import shutil
from pathlib import Path
from browsergym.experiments import StepInfo


def find_env_errors(trajectories_dir, output_file=None, move_to_dir=None, error_map_file=None):
    """
    查找所有任务轨迹中非空的 last_action_error，以及相关的 action 和 thought

    Args:
        trajectories_dir: 包含所有任务轨迹文件夹的目录路径
        output_file: 输出结果的JSON文件路径
        move_to_dir: 移动含有错误的任务文件夹到此目录
        error_map_file: 错误转换映射文件的路径
    """
    results = []
    # 用于存储错误映射的字典
    error_transition_map = {}
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

        # 标记是否在此任务中找到了错误
        task_has_error = False

        for i, step_file in enumerate(step_files):
            step_num = int(os.path.basename(step_file).split("_")[1].split(".")[0])
            step_name = f"step_{step_num}"

            try:
                with gzip.open(step_file, "rb") as f:
                    step_data = pickle.load(f)

                # 使用StepInfo类来处理step数据
                step_info = step_data
                if not isinstance(step_data, StepInfo):
                    print(f"警告: {step_file} 中的数据不是StepInfo类型")
                    continue

                # 从StepInfo对象中获取last_action_error
                error_message = step_info.obs.get("last_action_error", "")

                if "TimeoutError: Locator.click: Timeout 500ms exceeded." in error_message:
                    continue

                # 只有当错误消息不为空时才添加到结果中
                if error_message:
                    task_has_error = True

                    error_entry = {
                        "task_name": task_name,
                        "current_step": step_name,
                        "previous_step_error": error_message,
                        "previous_step_action": None,
                        "previous_step_thought": None,
                        "current_step_action": None,
                        "current_step_thought": None,
                    }

                    # 更新错误映射
                    if task_name not in error_transition_map:
                        error_transition_map[task_name] = []
                    error_transition_map[task_name].append(step_name)

                    # 查找上一步（导致错误的步骤）的action和thought
                    if step_num > 0:
                        prev_step_file = os.path.join(task_dir, f"step_{step_num-1}.pkl.gz")
                        if os.path.exists(prev_step_file):
                            try:
                                with gzip.open(prev_step_file, "rb") as f:
                                    prev_step_data = pickle.load(f)

                                if isinstance(prev_step_data, StepInfo):
                                    # 获取导致错误的action
                                    error_step_action = prev_step_data.action
                                    error_entry["previous_step_action"] = error_step_action

                                    # 获取导致错误的thought
                                    if (
                                        hasattr(prev_step_data, "agent_info")
                                        and prev_step_data.agent_info
                                    ):
                                        if hasattr(prev_step_data.agent_info, "think"):
                                            error_entry["previous_step_thought"] = (
                                                prev_step_data.agent_info.think
                                            )
                                        elif (
                                            isinstance(prev_step_data.agent_info, dict)
                                            and "think" in prev_step_data.agent_info
                                        ):
                                            error_entry["previous_step_thought"] = (
                                                prev_step_data.agent_info["think"]
                                            )
                            except Exception as e:
                                print(f"处理上一步文件 {prev_step_file} 时出错: {str(e)}")

                    # 获取当前步的action和thought（观察到错误后的反应）
                    if hasattr(step_info, "action"):
                        error_entry["current_step_action"] = step_info.action

                    if hasattr(step_info, "agent_info") and step_info.agent_info:
                        if hasattr(step_info.agent_info, "think"):
                            error_entry["current_step_thought"] = step_info.agent_info.think
                        elif (
                            isinstance(step_info.agent_info, dict)
                            and "think" in step_info.agent_info
                        ):
                            error_entry["current_step_thought"] = step_info.agent_info["think"]

                    results.append(error_entry)
                    print(f"找到错误 - 任务: {task_name}, 步骤: {step_name}, 错误: {error_message}")

            except Exception as e:
                print(f"处理文件 {step_file} 时出错: {str(e)}")

        # 如果该任务有错误且需要移动文件夹
        if task_has_error and move_to_dir and task_name not in moved_tasks:
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

    # 如果提供了输出文件路径，将结果保存为JSON
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到 {output_file}")
        except Exception as e:
            print(f"保存结果到文件时出错: {str(e)}")

    # 如果提供了错误映射文件路径，将错误映射保存为JSON
    if error_map_file:
        try:
            with open(error_map_file, "w", encoding="utf-8") as f:
                json.dump(error_transition_map, f, ensure_ascii=False, indent=2)
            print(f"错误转换映射已保存到 {error_map_file}")
        except Exception as e:
            print(f"保存错误映射到文件时出错: {str(e)}")

    return results, error_transition_map


if __name__ == "__main__":
    # 设置轨迹目录路径
    trajectories_dir = (
        "/home/weichenzhang/hallucination/AL_for_hallucination/src/agentlab/agents/test_results"
    )

    # 设置输出文件路径
    output_file = os.path.join(trajectories_dir, "env_errors_analysis.json")

    # 设置移动目标目录
    move_to_dir = (
        "/home/weichenzhang/hallucination/AL_for_hallucination/hallucination_traj_sample/WorkArena"
    )

    # 错误映射文件路径
    error_map_file = "/home/weichenzhang/hallucination/AL_for_hallucination/hallucination_traj_sample/WorkArena/error_transition_truncate_map.json"

    # 确保目标目录存在
    os.makedirs(move_to_dir, exist_ok=True)

    # 如果目录存在就执行查找
    if os.path.exists(trajectories_dir):
        errors, error_map = find_env_errors(
            trajectories_dir, output_file, move_to_dir, error_map_file
        )

        print("\n所有非空 last_action_error 分析结果:")
        print("=" * 80)
        for entry in errors:
            print(f"任务: {entry['task_name']}")
            print(f"错误步骤: {entry['current_step']}")
            print(f"错误: {entry['previous_step_error']}")
            print(f"导致错误的action: {entry['previous_step_action']}")
            print(f"导致错误的thought: {entry['previous_step_thought']}")
            print(f"观察到错误后的action: {entry['current_step_action']}")
            print(f"观察到错误后的thought: {entry['current_step_thought']}")
            print("-" * 80)

        print(f"总共找到 {len(errors)} 个错误")
        print(f"详细分析结果已保存到 {output_file}")
        print(f"错误转换映射已保存到 {error_map_file}")
        print(f"包含错误的任务已复制到 {move_to_dir}")
    else:
        print(f"错误: 目录 {trajectories_dir} 不存在")
