from browsergym.experiments.loop import ExpResult # 确保这个库已安装且路径正确
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
            elif type(obj) not in (str, int, float, bool, list, dict, type(None)):
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


def format_assistant_content_for_history(action_content: Any, step_idx: int) -> str:
    """将助手的 content (可能是字符串或字典) 格式化为历史记录中的文本。"""
    thought_action_text = ""
    if isinstance(action_content, str):
        # 假设字符串格式已经是 "<think>...</think>\n<action>...</action>"
        thought_action_text = action_content
    elif isinstance(action_content, dict):
        think_text = action_content.get("think", action_content.get("thought", "")) # 兼容 "think" 和 "thought"
        act_text = action_content.get("action", "")
        # 确保即使思考为空，标签也存在，以匹配示例格式
        formatted_think = f"<think>\n{think_text}\n</think>"
        formatted_action = f"<action>\n{act_text}\n</action>"
        thought_action_text = f"{formatted_think}\n\n{formatted_action}"
    
    if thought_action_text:
        return f"\n\n## step {step_idx}\n\n{thought_action_text}"
    return ""


def get_trajectory_and_simulate_failure(exp_dir: str, fail_step_idx: int):
    """
    构建一个单一的 User 消息，其 content 包含：
    - 第 n 步原始提示中的指令、目标、动作空间等。
    - 第 n 步原始提示中的观察（即 Obs_N，执行动作 N 前的观察）。
    - 更新后的历史，包含截至 n-1 步的历史，并追加第 n 步的思考和动作。
    最终的 input 序列是 [System_N, Single_User_Message_Simulated_N_Plus_1_State]
    """
    
    fail_step_file = os.path.join(exp_dir, f"step_{fail_step_idx}.pkl.gz")
    if not os.path.exists(fail_step_file):
        return None, f"目标步骤文件 step_{fail_step_idx}.pkl.gz 未找到。"
    fail_step_info = load_step_info(fail_step_file)
    if fail_step_info is None:
        return None, f"无法加载目标步骤 step_{fail_step_idx}.pkl.gz 的信息。"

    raw_chat_messages_for_step_n = []
    if hasattr(fail_step_info, "agent_info") and fail_step_info.agent_info and \
       hasattr(fail_step_info.agent_info, "chat_messages") and fail_step_info.agent_info.chat_messages:
        raw_chat_messages_for_step_n = fail_step_info.agent_info.chat_messages
    else:
        return None, f"步骤 {fail_step_idx} 的 agent_info.chat_messages 不可用或为空。"

    if not raw_chat_messages_for_step_n:
         return None, f"步骤 {fail_step_idx} 的 agent_info.chat_messages 为空。"

    # 1. 分离出第n步的提示消息 (system, user) 和第n步的助手动作消息
    prompt_messages_for_nth_action = []
    nth_action_assistant_message = None

    if raw_chat_messages_for_step_n[-1].get("role") == "assistant":
        nth_action_assistant_message = raw_chat_messages_for_step_n[-1]
        prompt_messages_for_nth_action = raw_chat_messages_for_step_n[:-1]
    else:
        prompt_messages_for_nth_action = raw_chat_messages_for_step_n
        print(f"警告: 步骤 {fail_step_idx} 的 chat_messages 未以助手消息结尾。无法提取第n步的动作以嵌入历史。")

    # 2. 从提示消息中提取第一个 system 消息和最后一个 user 消息
    selected_system_message = None
    for msg in prompt_messages_for_nth_action:
        if msg.get("role") == "system":
            if selected_system_message is None:
                selected_system_message = msg
            break 

    original_user_prompt_for_nth_step = None # 这是完整的 user 消息对象
    for msg in reversed(prompt_messages_for_nth_action):
        if msg.get("role") == "user":
            original_user_prompt_for_nth_step = msg
            break
    
    if not original_user_prompt_for_nth_step:
        return None, f"步骤 {fail_step_idx} 的提示中未找到用户消息。"

    original_user_content_list = original_user_prompt_for_nth_step.get("content")
    if not isinstance(original_user_content_list, list):
        # 如果 user content 不是预期的列表格式，则无法按块处理
        print(f"警告: 步骤 {fail_step_idx} 的用户消息 content 不是列表格式。将尝试作为单一文本处理。")
        # 这种情况下，无法精确更新历史或保持观察，模拟可能不准确
        # 但我们可以尝试将整个 content 视为一个整体的“观察+历史”块
        # 并尝试追加第n步的动作文本
        current_obs_plus_context_text = ""
        if isinstance(original_user_content_list, str):
            current_obs_plus_context_text = original_user_content_list
        else: # 其他类型，尝试str转换
            current_obs_plus_context_text = str(original_user_content_list)
        
        formatted_nth_action_text = ""
        if nth_action_assistant_message:
            formatted_nth_action_text = format_assistant_content_for_history(
                nth_action_assistant_message.get("content"), fail_step_idx
            )
        
        # 构建单个 user 消息的 content
        # 注意：这里的 current_obs_plus_context_text 已经包含了第n步执行前的观察 Obs_N
        final_user_text = current_obs_plus_context_text + formatted_nth_action_text
        modified_user_content_list = [{"type": "text", "text": final_user_text}]

    else: # user content 是列表，按块处理
        modified_user_content_list = []
        history_marker_lstrip = "# History of interaction with the task:"
        # observation_marker_lstrip = "# Observation of current step:" # 此块内容保持不变

        formatted_nth_action_text = ""
        if nth_action_assistant_message:
            formatted_nth_action_text = format_assistant_content_for_history(
                nth_action_assistant_message.get("content"), fail_step_idx
            )

        history_block_updated = False
        for content_block_dict in original_user_content_list:
            block_copy = dict(content_block_dict) # 创建副本以修改
            original_text = block_copy.get("text", "")

            # 我们主要关心更新历史记录块
            if original_text.lstrip().startswith(history_marker_lstrip):
                block_copy["text"] = original_text.rstrip() + formatted_nth_action_text # 追加第n步动作到历史
                history_block_updated = True
            
            # 其他块 (包括观察块 Observation_N) 直接沿用原始第n步提示中的内容
            modified_user_content_list.append(block_copy)

        # 如果原始提示中没有历史块，但我们有第n步的动作，则需要添加一个新的历史块
        if not history_block_updated and formatted_nth_action_text:
            print(f"警告: 步骤 {fail_step_idx} 的原始用户提示中未找到历史记录块。将创建新的历史记录块。")
            # 通常历史记录块前面会有一些引导文本，这里简化处理
            history_intro_text = "\n# History of interaction with the task:"
            new_history_block_text = history_intro_text + formatted_nth_action_text
            modified_user_content_list.append({"type": "text", "text": new_history_block_text})


    # 3. 构建最终的 input 序列: [S_N, Single_User_Message]
    full_input_sequence = []
    if selected_system_message:
        full_input_sequence.append(selected_system_message)
    
    full_input_sequence.append({"role": "user", "content": modified_user_content_list})
             
    return full_input_sequence, None


def safe_extract(obj: Any, attr_path: str, default=None):
    attrs = attr_path.split(".")
    current = obj
    for attr in attrs:
        if hasattr(current, attr):
            current = getattr(current, attr)
        elif isinstance(current, dict) and attr in current:
            current = current[attr]
        else:
            return default
    return current


def extract_info_from_exp_args(exp_args):
    if exp_args is None:
        print("警告: exp_args 为 None，无法提取信息。")
        return {"error": "exp_args is None, failed to extract info."}
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


def _generate_single_json_for_failed_step(exp_dir: str, exp_args: Any, options: Any, fail_step_num: int, suffix: str = ""):
    info = extract_info_from_exp_args(exp_args)
    if "error" in info and info.get("error"):
        print(f"无法从 {exp_dir} 的 exp_args 提取基础信息。错误: {info.get('error')}")
    
    info["input_step"] = fail_step_num 

    if options.include_goal:
        goal_object = load_goal_object(exp_dir)
        if goal_object:
            info["goal"] = str(goal_object) 
        else:
            info["goal"] = "Goal object could not be loaded."
    else:
        info["goal"] = "Goal not included by user option."

    input_content, error_msg = get_trajectory_and_simulate_failure(exp_dir, fail_step_num)
    if input_content:
        info["input"] = input_content
    else:
        error_detail = error_msg if error_msg else "未知错误"
        print(f"无法为步骤 {fail_step_num} 获取模拟失败的轨迹: {error_detail}")
        info["input_error"] = f"获取轨迹失败 (步骤 {fail_step_num}): {error_detail}"

    base_output_name = options.output_filename
    if suffix:
        name_part, ext_part = os.path.splitext(base_output_name)
        output_filename = f"{name_part}{suffix}{ext_part}"
    else:
        output_filename = base_output_name
    
    exp_dir_name = os.path.basename(exp_dir)
    risk_keywords = ["unreachable", "misleading", "ambiguity", "missinginfo", "human_in_loop"]
    risk_type = next((kw for kw in risk_keywords if kw.lower() in exp_dir_name.lower()), None)

    if risk_type:
         output_filename = f"{risk_type}_{output_filename}"

    info_path = os.path.join(exp_dir, output_filename)

    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False, cls=CustomEncoder)
        print(f"已生成 {info_path}")
        return True
    except Exception as e:
        print(f"保存 {output_filename} 时出错: {e}")
        try:
            simple_info_to_save = {}
            keys_to_try = ["task_name", "task_seed", "agent_name", "input_step", "goal", "input_error", "model_config", "max_steps", "observation_settings"]
            for key in keys_to_try:
                if key in info and isinstance(info[key], (str, int, float, bool, list, dict, type(None))):
                    simple_info_to_save[key] = info[key]
                elif key in info:
                     simple_info_to_save[key] = f"<Data for '{key}' was not simple string/number/etc>"
            simple_info_to_save["error_saving_full_json"] = f"完整信息无法序列化: {str(e)}"
            if "input" in info and not isinstance(info["input"], list): 
                simple_info_to_save["input_type_issue"] = f"Input field was of type {type(info['input']).__name__}, expected list."

            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(simple_info_to_save, f, indent=4, ensure_ascii=False, cls=CustomEncoder)
            print(f"已生成简化版 {info_path} (由于序列化错误)")
            return True
        except Exception as e2:
            print(f"保存简化版 {output_filename} 时也出错: {e2}")
            return False


def generate_all_info_jsons_for_exp(exp_dir, exp_args, options, truncate_map):
    exp_dir_name = os.path.basename(exp_dir)
    fail_steps_data = truncate_map.get(exp_dir_name)
    if fail_steps_data is None:
        return 0 

    generated_files_count = 0
    if isinstance(fail_steps_data, list):
        for fail_step_num in fail_steps_data:
            if not isinstance(fail_step_num, int) or fail_step_num < 0:
                print(f"警告: {exp_dir_name} 的截断映射包含无效步骤 {fail_step_num}。跳过。")
                continue
            suffix = f"_FS{fail_step_num}"
            if _generate_single_json_for_failed_step(exp_dir, exp_args, options, fail_step_num, suffix):
                generated_files_count += 1
    elif isinstance(fail_steps_data, int) and fail_steps_data >= 0:
        fail_step_num = fail_steps_data
        suffix = f"_FS{fail_step_num}"
        if _generate_single_json_for_failed_step(exp_dir, exp_args, options, fail_step_num, suffix):
            generated_files_count += 1
    else:
        print(f"警告: {exp_dir_name} 的截断映射条目格式无效: {fail_steps_data}。")
    return generated_files_count


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
    parser = argparse.ArgumentParser(description="生成包含模拟失败动作轨迹的实验信息JSON文件。")
    parser.add_argument("--dir", type=str, help="指定要处理的实验目录（或包含实验目录的父目录）。默认为脚本所在目录。")
    parser.add_argument(
        "--output", type=str, default="simulated_failed_input.json",
        help="输出JSON文件的基础名称（后缀如 _FS<N>.json 会被添加）。默认为 simulated_failed_input.json"
    )
    parser.add_argument("--quiet", action="store_true", help="减少输出信息")
    parser.add_argument("--no-goal", action="store_true", help="不包含目标对象信息")
    parser.add_argument(
        "--truncate-map",
        type=str,
        default="truncate_map.json",
        help="JSON文件路径，指定每个实验要模拟失败的步骤索引（0-indexed）。"
             "示例: {\"exp_dir1\": 5, \"exp_dir2\": [0, 3]}"
    )
    args = parser.parse_args()
    class Options:
        def __init__(self, parsed_args):
            self.dir = parsed_args.dir
            self.quiet = parsed_args.quiet
            self.include_goal = not parsed_args.no_goal
            self.output_filename = parsed_args.output
            self.truncate_map_file = parsed_args.truncate_map
    return Options(args)


def main():
    options = parse_args()
    if options.dir:
        base_search_path = Path(options.dir).resolve()
    else:
        base_search_path = Path(__file__).parent.resolve()

    truncate_map_path = Path(options.truncate_map_file)
    if not truncate_map_path.is_absolute():
        truncate_map_path = Path.cwd() / options.truncate_map_file
    
    truncate_map = load_truncate_map(truncate_map_path)
    if truncate_map:
        print(f"已加载截断映射 '{truncate_map_path}', 包含 {len(truncate_map)} 个实验目录的映射")
    else:
        print(f"警告: 未找到或无法加载截断映射文件 {truncate_map_path} 或文件为空。")
        truncate_map = {}

    exp_dirs_paths = []
    if is_experiment_dir(base_search_path):
        exp_dirs_paths = [base_search_path]
        if not options.quiet: print(f"--dir '{base_search_path}' 本身是一个实验目录。")
    else:
        if not options.quiet: print(f"在 '{base_search_path}' 中查找实验子目录...")
        for d_path in base_search_path.iterdir():
            if d_path.is_dir() and is_experiment_dir(d_path):
                 exp_dirs_paths.append(d_path)
    
    if not exp_dirs_paths and not options.quiet :
        print(f"在 '{base_search_path}' 中未找到实验目录。请检查 --dir 参数或脚本位置。")

    print(f"找到 {len(exp_dirs_paths)} 个实验目录进行处理。")

    processed_exp_count = 0
    skipped_due_to_map_count = 0
    total_files_generated = 0
    failed_to_load_args_count = 0

    for exp_dir_path_obj in exp_dirs_paths:
        exp_dir = str(exp_dir_path_obj)
        exp_dir_name = exp_dir_path_obj.name

        if exp_dir_name not in truncate_map:
            if not options.quiet:
                print(f"\n跳过 {exp_dir_name}：不在截断映射 '{truncate_map_path}' 中。")
            skipped_due_to_map_count += 1
            continue

        if not options.quiet:
            print("\n" + "=" * 80)
            print(f"实验目录: {exp_dir_name}")
            print("=" * 80)
            fail_steps_data = truncate_map[exp_dir_name]
            print(f"将为步骤索引模拟失败: {fail_steps_data}")
        else:
            print(f"处理: {exp_dir_name}, 模拟失败步骤: {truncate_map.get(exp_dir_name)}")

        exp_args = load_exp_args(exp_dir)
        if exp_args:
            if not options.quiet:
                print("\n重要信息摘要:")
                print(f"  任务名称: {safe_extract(exp_args, 'env_args.task_name', 'N/A')}")
                print(f"  任务种子: {safe_extract(exp_args, 'env_args.task_seed', 'N/A')}")
                print(f"  智能体名称: {safe_extract(exp_args, 'agent_args.agent_name', 'N/A')}")
            
            files_generated_for_this_exp = generate_all_info_jsons_for_exp(exp_dir, exp_args, options, truncate_map)
            if files_generated_for_this_exp > 0:
                processed_exp_count += 1
                total_files_generated += files_generated_for_this_exp
            elif exp_dir_name in truncate_map and not options.quiet:
                 print(f"注意: {exp_dir_name} 在映射中，但没有为其生成文件。")

            if not options.quiet:
                summary = load_result_summary(exp_dir)
                if summary:
                    print("\n原始实验结果摘要 (供参考):")
                    print(f"  累计奖励: {summary.get('cum_reward', 'N/A')}")
                    print(f"  步数: {summary.get('n_steps', 'N/A')}")
                    print(f"  是否成功: {summary.get('success', 'N/A')}")
        else:
            print(f"无法加载 {exp_dir_name} 的实验参数，已跳过此目录的JSON生成。")
            failed_to_load_args_count +=1

    print(f"\n--- 处理总结 ---")
    print(f"成功为其生成至少一个文件的实验目录数: {processed_exp_count}")
    print(f"总共生成的模拟输入文件数: {total_files_generated}")
    print(f"由于不在截断映射中而跳过的实验目录数: {skipped_due_to_map_count}")
    if failed_to_load_args_count > 0:
        print(f"由于加载exp_args失败而未能处理的实验目录数: {failed_to_load_args_count}")
    
    total_considered = len(exp_dirs_paths)
    unaccounted_for = total_considered - processed_exp_count - skipped_due_to_map_count - failed_to_load_args_count
    if unaccounted_for > 0 and not options.quiet:
         print(f"注意: 有 {unaccounted_for} 个实验目录可能由于其他原因未被完全处理。")

if __name__ == "__main__":
    main()