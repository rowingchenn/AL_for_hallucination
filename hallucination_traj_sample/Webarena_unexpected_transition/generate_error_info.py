from browsergym.experiments.loop import ExpResult
import os
from pathlib import Path
import pickle
import gzip
import json
from pprint import pprint
import re
import argparse
from typing import Any, Dict, List, Union


class CustomEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理不可序列化的对象"""

    def default(self, obj):
        try:
            if hasattr(obj, "__dict__"):
                return {
                    key: value for key, value in obj.__dict__.items() if not key.startswith("_")
                }
            elif hasattr(obj, "tolist"):  # 处理numpy数组
                return obj.tolist()
            elif hasattr(obj, "__str__"):
                return str(obj)
            return json.JSONEncoder.default(self, obj)
        except Exception:
            return f"<无法序列化的对象: {type(obj).__name__}>"


def is_experiment_dir(path):
    """判断一个目录是否为实验记录目录"""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "exp_args.pkl"))


def load_exp_args(exp_dir):
    """从实验目录加载exp_args.pkl"""
    try:
        # 使用ExpResult加载
        exp_result = ExpResult(exp_dir)
        return exp_result.exp_args
    except Exception as e:
        print(f"无法使用ExpResult加载 {exp_dir}: {e}")

        # 回退方法：直接使用pickle加载
        try:
            exp_args_path = os.path.join(exp_dir, "exp_args.pkl")
            with open(exp_args_path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"无法直接使用pickle加载 {exp_args_path}: {e2}")
            return None


def load_goal_object(exp_dir):
    """加载目标对象信息"""
    try:
        goal_path = os.path.join(exp_dir, "goal_object.pkl.gz")
        if os.path.exists(goal_path):
            with gzip.open(goal_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"加载目标对象时出错: {e}")
    return None


def load_step_info(step_path):
    """加载步骤信息"""
    try:
        with gzip.open(step_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"加载步骤信息时出错: {step_path}: {e}")
    return None


def load_truncate_map(map_path):
    """加载截断映射信息"""
    try:
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"加载截断映射信息时出错: {e}")
    return {}


def get_input_from_step(exp_dir, step_num):
    """从指定步骤中提取输入信息"""
    step_file = os.path.join(exp_dir, f"step_{step_num}.pkl.gz")
    if not os.path.exists(step_file):
        return None, "步骤文件不存在"

    step_info = load_step_info(step_file)
    if step_info is None:
        return None, "无法加载步骤信息"

    try:
        chat_messages = step_info.agent_info.chat_messages
        # 找到最前面出现 role = assistant 的位置
        assistant_index = -1
        for i, msg in enumerate(chat_messages):
            if msg.get("role") == "assistant":
                assistant_index = i
                break

        # 如果找到了assistant消息，则取[:assistant_index]，否则保持原样
        if assistant_index != -1:
            chat_messages = chat_messages[:assistant_index]

        return chat_messages, None  # 添加成功时的返回语句
    except Exception as e:
        return None, f"提取输入信息时出错: {str(e)}"


def get_action_from_step(exp_dir, step_num):
    """从指定步骤中提取action信息"""
    step_file = os.path.join(exp_dir, f"step_{step_num}.pkl.gz")
    if not os.path.exists(step_file):
        return None, "步骤文件不存在"

    step_info = load_step_info(step_file)
    if step_info is None:
        return None, "无法加载步骤信息"

    # 尝试提取action信息
    try:
        if hasattr(step_info, "action"):
            return step_info.action, None
        return None, "在步骤信息中未找到action数据"
    except Exception as e:
        return None, f"提取action信息时出错: {str(e)}"


def safe_extract(obj: Any, attr_path: str, default=None):
    """安全地从对象中提取嵌套属性"""
    attrs = attr_path.split(".")
    current = obj

    for attr in attrs:
        if hasattr(current, attr):
            current = getattr(current, attr)
        else:
            return default

    return current


def extract_info_from_exp_args(exp_args):
    """从实验参数中提取关键信息"""
    try:
        # 提取基本信息
        info = {
            "task_name": safe_extract(exp_args, "env_args.task_name"),
            "task_seed": safe_extract(exp_args, "env_args.task_seed"),
            "agent_name": safe_extract(exp_args, "agent_args.agent_name"),
            "model_config": {
                "model_name": safe_extract(exp_args, "agent_args.chat_model_args.model_name"),
                "max_tokens": safe_extract(exp_args, "agent_args.chat_model_args.max_new_tokens"),
                "temperature": safe_extract(exp_args, "agent_args.chat_model_args.temperature"),
            },
            "max_steps": safe_extract(exp_args, "env_args.max_steps"),
            "observation_settings": {
                "use_html": safe_extract(exp_args, "agent_args.flags.obs.use_html"),
                "use_ax_tree": safe_extract(exp_args, "agent_args.flags.obs.use_ax_tree"),
                "use_screenshot": safe_extract(exp_args, "agent_args.flags.obs.use_screenshot"),
                "use_som": safe_extract(exp_args, "agent_args.flags.obs.use_som"),
            },
        }

        return info
    except Exception as e:
        print(f"提取信息时出错: {e}")
        return {"error": f"提取信息失败: {str(e)}", "raw_exp_args": str(exp_args)}


def generate_info_json_for_step(exp_dir, exp_args, options, step_num, suffix=""):
    """为单个步骤生成并保存info.json文件"""
    # 基本信息
    info = extract_info_from_exp_args(exp_args)
    if not info:
        print(f"无法从 {exp_dir} 提取信息")
        return False

    # 加载目标对象
    if options.include_goal:
        goal_object = load_goal_object(exp_dir)
        if goal_object:
            info["goal"] = str(goal_object)

    # 添加输入步骤和输入内容
    info["input_step"] = step_num

    input_content, error = get_input_from_step(exp_dir, step_num)
    if input_content:
        info["input"] = input_content
    elif error:
        print(f"无法获取步骤 {step_num} 的输入内容: {error}")

    # 确定输出文件名
    if suffix:
        output_filename = options.output_filename.replace("_info.json", f"{suffix}_info.json")
    else:
        output_filename = options.output_filename

    # 保存到info.json
    info_path = os.path.join(exp_dir, output_filename)
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False, cls=CustomEncoder)

        print(f"已生成 {info_path}")
        return True
    except Exception as e:
        print(f"保存 {output_filename} 时出错: {e}")

        # 尝试保存简化版本
        try:
            simple_info = {
                "task_name": info.get("task_name"),
                "task_seed": info.get("task_seed"),
                "agent_name": info.get("agent_name"),
                "input_step": step_num,
                "error": f"完整信息无法序列化: {str(e)}",
            }
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(simple_info, f, indent=4, ensure_ascii=False)
            print(f"已生成简化版 {info_path}")
            return True
        except Exception as e2:
            print(f"保存简化版 {output_filename} 时也出错: {e2}")
            return False


def generate_info_json(exp_dir, exp_args, options, truncate_map):
    """生成并保存info.json文件，处理单个步骤或多个步骤的情况"""
    exp_dir_name = os.path.basename(exp_dir)

    # 获取映射中的值
    steps_data = truncate_map.get(exp_dir_name)
    if steps_data is None:
        print(f"错误: {exp_dir_name} 不在截断映射中")
        return False

    # 判断是单个步骤还是多个步骤
    if isinstance(steps_data, list):
        # 多个步骤的情况
        # 过滤连续的相同action步骤，确保连续不超过2个
        filtered_steps = []
        last_action = None
        last_action_str = None
        consecutive_count = 0

        for step_num in sorted(steps_data):
            action, error = get_action_from_step(exp_dir, step_num)
            action_str = str(action) if action is not None else None

            if action_str == last_action_str:
                consecutive_count += 1
                if consecutive_count <= 2:  # 允许最多连续2个相同action
                    filtered_steps.append(step_num)
            else:
                consecutive_count = 1
                filtered_steps.append(step_num)

            last_action_str = action_str

        if not options.quiet:
            if len(filtered_steps) < len(steps_data):
                print(f"原始步骤: {steps_data}")
                print(
                    f"过滤后步骤: {filtered_steps}（已移除{len(steps_data) - len(filtered_steps)}个连续相同action的步骤）"
                )
            else:
                print(f"所有步骤都保留: {filtered_steps}")

        # 使用过滤后的步骤
        success_count = 0
        for step_num in filtered_steps:
            # 生成带有步骤编号的后缀
            suffix = f"_S{step_num}"
            if generate_info_json_for_step(exp_dir, exp_args, options, step_num, suffix):
                success_count += 1

        return success_count > 0
    else:
        # 单个步骤的情况（兼容旧格式）
        step_num = steps_data
        return generate_info_json_for_step(exp_dir, exp_args, options, step_num)


def load_result_summary(exp_dir):
    """尝试加载实验结果摘要"""
    try:
        summary_path = os.path.join(exp_dir, "summary_info.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"加载摘要信息时出错: {e}")
    return None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成实验信息JSON文件")
    parser.add_argument("--dir", type=str, help="指定要处理的实验目录，默认为当前目录")
    parser.add_argument(
        "--output", type=str, default="info.json", help="输出文件名，默认为info.json"
    )
    parser.add_argument("--quiet", action="store_true", help="减少输出信息")
    parser.add_argument("--no-goal", action="store_true", help="不包含目标对象信息")
    parser.add_argument(
        "--truncate-map",
        type=str,
        default="error_transition_truncate_map.json",
        help="截断映射文件，指定每个实验的关键输入步骤",
    )

    args = parser.parse_args()

    # 转换为选项对象
    class Options:
        def __init__(self, args):
            self.dir = args.dir
            self.quiet = args.quiet
            self.include_goal = not args.no_goal
            self.output_filename = args.output
            self.truncate_map_file = args.truncate_map

    return Options(args)


def main():
    # 解析命令行参数
    options = parse_args()

    # 获取当前目录或指定目录
    if options.dir:
        current_dir = Path(options.dir)
    else:
        current_dir = Path(__file__).parent

    # 加载截断映射
    truncate_map_path = os.path.join(current_dir, options.truncate_map_file)
    truncate_map = load_truncate_map(truncate_map_path)
    if truncate_map:
        print(f"已加载截断映射，包含 {len(truncate_map)} 个实验目录的映射")
    else:
        print(f"警告: 未找到截断映射文件 {options.truncate_map_file} 或文件为空")
        truncate_map = {}

    # 查找所有实验目录
    exp_dirs = [
        os.path.join(current_dir, d)
        for d in os.listdir(current_dir)
        if is_experiment_dir(os.path.join(current_dir, d))
    ]

    print(f"找到 {len(exp_dirs)} 个实验目录")

    success_count = 0
    skipped_count = 0
    total_files_generated = 0

    # 对于每个实验目录，加载并处理exp_args
    for exp_dir in exp_dirs:
        exp_dir_name = os.path.basename(exp_dir)

        # 检查是否在截断映射中
        if exp_dir_name not in truncate_map:
            if not options.quiet:
                print(f"\n跳过 {exp_dir_name}：不在截断映射中")
            skipped_count += 1
            continue

        if not options.quiet:
            print("\n" + "=" * 80)
            print(f"实验目录: {exp_dir_name}")
            print("=" * 80)

            # 显示关键输入步骤（可能是单个步骤或数组）
            steps_data = truncate_map[exp_dir_name]
            if isinstance(steps_data, list):
                print(f"关键输入步骤: {steps_data} (多个步骤)")
            else:
                print(f"关键输入步骤: {steps_data}")
        else:
            print(f"处理: {exp_dir_name}")

        exp_args = load_exp_args(exp_dir)
        if exp_args:
            # 打印实验参数
            if not options.quiet:
                print("实验参数:")
                pprint(exp_args)

                # 打印一些重要信息
                try:
                    print("\n重要信息摘要:")
                    print(f"任务名称: {safe_extract(exp_args, 'env_args.task_name')}")
                    print(f"任务种子: {safe_extract(exp_args, 'env_args.task_seed')}")
                    print(f"智能体名称: {safe_extract(exp_args, 'agent_args.agent_name')}")
                except Exception as e:
                    print(f"无法提取某些重要信息: {e}")

            # 生成info.json
            result = generate_info_json(exp_dir, exp_args, options, truncate_map)
            if result:
                success_count += 1
                # 计算生成的文件数量
                if isinstance(truncate_map[exp_dir_name], list):
                    # 这里需要修改，因为我们现在过滤了一些步骤
                    # 查找当前目录下以特定后缀命名的文件数量来计算实际生成的文件数量
                    pattern = options.output_filename.replace(".json", "_S*.json")
                    matching_files = [
                        f
                        for f in os.listdir(exp_dir)
                        if f.startswith(os.path.basename(pattern.replace("*", "")))
                    ]
                    total_files_generated += len(matching_files)
                else:
                    total_files_generated += 1

            # 尝试加载结果摘要(仅显示，不写入info.json)
            if not options.quiet:
                summary = load_result_summary(exp_dir)
                if summary:
                    print("\n结果摘要:")
                    print(f"累计奖励: {summary.get('cum_reward', 'N/A')}")
                    print(f"步数: {summary.get('n_steps', 'N/A')}")
                    print(f"是否成功: {summary.get('success', 'N/A')}")
        else:
            print("无法加载实验参数")

    print(f"\n成功处理 {success_count}/{len(exp_dirs) - skipped_count} 个实验目录")
    print(f"总共生成了 {total_files_generated} 个info文件")
    print(f"跳过 {skipped_count} 个不在截断映射中的实验目录")


if __name__ == "__main__":
    main()
