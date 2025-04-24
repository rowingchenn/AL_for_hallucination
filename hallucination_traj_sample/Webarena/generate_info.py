from browsergym.experiments.loop import ExpResult
import os
from pathlib import Path
import pickle
import gzip
import json
from pprint import pprint
import re
import argparse
from typing import Any, Dict


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
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "exp_args.pkl"))

def load_exp_args(exp_dir):
    try:
        exp_result = ExpResult(exp_dir)
        return exp_result.exp_args
    except Exception as e:
        print(f"无法使用ExpResult加载 {exp_dir}: {e}")
        try:
            exp_args_path = os.path.join(exp_dir, "exp_args.pkl")
            with open(exp_args_path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"无法直接使用pickle加载 {exp_args_path}: {e2}")
            return None

def load_goal_object(exp_dir):
    try:
        goal_path = os.path.join(exp_dir, "goal_object.pkl.gz")
        if os.path.exists(goal_path):
            with gzip.open(goal_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"加载目标对象时出错: {e}")
    return None

def load_step_info(step_path):
    try:
        with gzip.open(step_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"加载步骤信息时出错: {step_path}: {e}")
    return None

def load_truncate_map(map_path):
    try:
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"加载截断映射信息时出错: {e}")
    return {}

def get_input_from_step(exp_dir, step_num):
    if step_num == 1:
        step_num = 0  # 对于 n=1 的情况，回退到 step_0

    step_file = os.path.join(exp_dir, f"step_{step_num}.pkl.gz")
    if not os.path.exists(step_file):
        return None, "步骤文件不存在"

    step_info = load_step_info(step_file)
    if step_info is None:
        return None, "无法加载步骤信息"

    try:
        if hasattr(step_info, "agent_info") and hasattr(step_info.agent_info, "chat_messages"):
            chat_messages = step_info.agent_info.chat_messages
            if len(chat_messages) > 0:
                for msg in reversed(chat_messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content"), None
                    elif hasattr(msg, "type") and (msg.type == "human" or msg.type == "user"):
                        return msg.content, None

        if hasattr(step_info, "obs"):
            for key in ["goal_object", "task_description", "prompt"]:
                if key in step_info.obs:
                    return step_info.obs[key], None

        return None, "在步骤信息中未找到输入数据"
    except Exception as e:
        return None, f"提取输入信息时出错: {str(e)}"

def safe_extract(obj: Any, attr_path: str, default=None):
    attrs = attr_path.split(".")
    current = obj
    for attr in attrs:
        if hasattr(current, attr):
            current = getattr(current, attr)
        else:
            return default
    return current

def extract_info_from_exp_args(exp_args):
    try:
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

'''
def generate_info_json(exp_dir, exp_args, options, truncate_map):
    info = extract_info_from_exp_args(exp_args)
    if not info:
        print(f"无法从 {exp_dir} 提取信息")
        return False

    if options.include_goal:
        goal_object = load_goal_object(exp_dir)
        if goal_object:
            info["goal"] = str(goal_object)

    exp_dir_name = os.path.basename(exp_dir)
    if exp_dir_name in truncate_map:
        input_step = truncate_map[exp_dir_name]
        info["input_step"] = input_step

        input_content, error = get_input_from_step(exp_dir, input_step)
        if input_content:
            info["input"] = input_content
        elif error:
            print(f"无法获取输入内容: {error}")

    info_path = os.path.join(exp_dir, options.output_filename)
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False, cls=CustomEncoder)
        print(f"已生成 {info_path}")
        return True
    except Exception as e:
        print(f"保存info.json时出错: {e}")
        try:
            simple_info = {
                "task_name": info.get("task_name"),
                "task_seed": info.get("task_seed"),
                "agent_name": info.get("agent_name"),
                "error": f"完整信息无法序列化: {str(e)}",
            }
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(simple_info, f, indent=4, ensure_ascii=False)
            print(f"已生成简化版 {info_path}")
            return True
        except Exception as e2:
            print(f"保存简化版info.json时也出错: {e2}")
            return False
'''

def generate_info_json(exp_dir, exp_args, options, truncate_map):
    info = extract_info_from_exp_args(exp_args)
    if not info:
        print(f"无法从 {exp_dir} 提取信息")
        return False

    if options.include_goal:
        goal_object = load_goal_object(exp_dir)
        if goal_object:
            info["goal"] = str(goal_object)

    exp_dir_name = os.path.basename(exp_dir)

    if exp_dir_name in truncate_map:
        input_step = truncate_map[exp_dir_name]
        info["input_step"] = input_step

        input_content, error = get_input_from_step(exp_dir, input_step)
        if input_content:
            info["input"] = input_content
        elif error:
            print(f"无法获取输入内容: {error}")

    # 构造输出文件名：<风险类型>_<目录名>.json
    risk_keywords = [
        "unreachable", "misleading", "ambiguity", "missinginfo", "human_in_loop", "unexpected_transition"
    ]
    risk_type = next((kw for kw in risk_keywords if kw in exp_dir_name), None)

    if risk_type:
        filename = f"{risk_type}_{exp_dir_name}.json"
    else:
        filename = f"{exp_dir_name}.json"

    info_path = os.path.join(exp_dir, filename)

    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False, cls=CustomEncoder)
        print(f"已生成 {info_path}")
        return True
    except Exception as e:
        print(f"保存info.json时出错: {e}")
        try:
            simple_info = {
                "task_name": info.get("task_name"),
                "task_seed": info.get("task_seed"),
                "agent_name": info.get("agent_name"),
                "error": f"完整信息无法序列化: {str(e)}",
            }
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(simple_info, f, indent=4, ensure_ascii=False)
            print(f"已生成简化版 {info_path}")
            return True
        except Exception as e2:
            print(f"保存简化版info.json时也出错: {e2}")
            return False


def load_result_summary(exp_dir):
    try:
        summary_path = os.path.join(exp_dir, "summary_info.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"加载摘要信息时出错: {e}")
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="生成实验信息JSON文件")
    parser.add_argument("--dir", type=str, help="指定要处理的实验目录，默认为当前目录")
    parser.add_argument("--output", type=str, default="info.json", help="输出文件名，默认为info.json")
    parser.add_argument("--quiet", action="store_true", help="减少输出信息")
    parser.add_argument("--no-goal", action="store_true", help="不包含目标对象信息")
    parser.add_argument("--truncate-map", type=str, default="truncate_map.json", help="截断映射文件，指定每个实验的关键输入步骤")
    args = parser.parse_args()

    class Options:
        def __init__(self, args):
            self.dir = args.dir
            self.quiet = args.quiet
            self.include_goal = not args.no_goal
            self.output_filename = args.output
            self.truncate_map_file = args.truncate_map

    return Options(args)

def main():
    options = parse_args()
    current_dir = Path(options.dir) if options.dir else Path(__file__).parent

    truncate_map_path = os.path.join(current_dir, options.truncate_map_file)
    truncate_map = load_truncate_map(truncate_map_path)
    if truncate_map:
        print(f"已加载截断映射，包含 {len(truncate_map)} 个实验目录的映射")
    else:
        print(f"警告: 未找到截断映射文件 {options.truncate_map_file} 或文件为空")
        truncate_map = {}

    exp_dirs = [
        os.path.join(current_dir, d)
        for d in os.listdir(current_dir)
        if is_experiment_dir(os.path.join(current_dir, d))
    ]

    print(f"找到 {len(exp_dirs)} 个实验目录")
    success_count = 0

    for exp_dir in exp_dirs:
        exp_dir_name = os.path.basename(exp_dir)

        if not options.quiet:
            print("\n" + "=" * 80)
            print(f"实验目录: {exp_dir_name}")
            print("=" * 80)
            if exp_dir_name in truncate_map:
                print(f"关键输入步骤: {truncate_map[exp_dir_name]}")
        else:
            print(f"处理: {exp_dir_name}")

        exp_args = load_exp_args(exp_dir)
        if exp_args:
            if not options.quiet:
                print("实验参数:")
                pprint(exp_args)
                try:
                    print("\n重要信息摘要:")
                    print(f"任务名称: {safe_extract(exp_args, 'env_args.task_name')}")
                    print(f"任务种子: {safe_extract(exp_args, 'env_args.task_seed')}")
                    print(f"智能体名称: {safe_extract(exp_args, 'agent_args.agent_name')}")
                except Exception as e:
                    print(f"无法提取某些重要信息: {e}")

            if generate_info_json(exp_dir, exp_args, options, truncate_map):
                success_count += 1

            if not options.quiet:
                summary = load_result_summary(exp_dir)
                if summary:
                    print("\n结果摘要:")
                    print(f"累计奖励: {summary.get('cum_reward', 'N/A')}")
                    print(f"步数: {summary.get('n_steps', 'N/A')}")
                    print(f"是否成功: {summary.get('success', 'N/A')}")
        else:
            print("无法加载实验参数")

    print(f"\n成功生成 {success_count}/{len(exp_dirs)} 个{options.output_filename}文件")

if __name__ == "__main__":
    main()
