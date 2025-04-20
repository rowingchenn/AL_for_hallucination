import bgym
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
import os
from tqdm import tqdm

# 获取当前 Python 文件的目录路径
current_directory = os.path.dirname(os.path.abspath(__file__))

# 将工作目录更改为当前文件目录
os.chdir(current_directory)

import numpy as np
import logging

logger = logging.getLogger(__name__)

os.environ["WA_SHOPPING"] = "http://111.229.174.217:18082"
os.environ["WA_SHOPPING_ADMIN"] = "http://111.229.174.217:18083/admin"
os.environ["WA_REDDIT"] = "http://111.229.174.217:19002"
os.environ["WA_GITLAB"] = "http://111.229.174.217:19001"
os.environ["WA_WIKIPEDIA"] = "http://111.229.174.217:19003"
os.environ["WA_MAP"] = "http://111.229.174.217:18084"
os.environ["WA_HOMEPAGE"] = "http://111.229.174.217:18081"
#os.environ["WA_FULL_RESET"] = "http://111.229.174.217:17565" # /satus 访问不了，因此暂时跳过
os.environ["WA_FULL_RESET"] = ""

FLAGS_TEST = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=False,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=bgym.HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
        long_description=True,
        individual_examples=True,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=False,
    use_hints=False,
    enable_chat=False,
    max_prompt_tokens=128000,  # The context of Qwen2.5-7B-Instruct is 128K tokens
    be_cautious=False,
    # extra_instructions="If you meet some problems and you can't solve, please don't do to many meaningless retries. Just report the problem to the user.",
    extra_instructions=None,
)

AGENT_TEST = GenericAgentArgs(
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o3-mini-2025-01-31"],
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-11-20"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o1-2024-12-17"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["google/gemini-2.0-flash"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o1-mini-2024-09-12"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4-1106-preview"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["local/Qwen2.5-7B-Instruct"],
    flags=FLAGS_TEST,
    max_retry=3,
)


def main():
    exp_dir = "./test_results/"

    # 定义要测试的任务列表
    task_name_list = [
        # "webarena.1",
        # "webarena.2",
        # "webarena.3", # 结果不对，且task还需要修改
        # "webarena.4", # 结果不对，且task还需要修改
        # "webarena.5", # task 需要修改，没有chatgpt-plugin这个repo
        # "webarena.6",
        # "webarena.7", # traj 不理想
        # "webarena.8", # 在 task 中没有要求agent翻遍完整窗口，当前agent好像只能看到window显示的部分，需要在instruction中显式说明请翻遍窗口？
        # "webarena.9",
        # "webarena.10",
        # "webarena.11", # 要手动修改traj，unexpected transition
        #  "webarena.12", # traj 不理想
        # "webarena.13",
        # "webarena.14",
        # "webarena.15",
        "webarena.16",
        "webarena.17",
        "webarena.18",
        "webarena.19"
    ]

    # 初始化空的实验参数列表
    exp_args_list = []

    # 循环生成每个任务的实验参数
    for task_name in task_name_list:
        env_args = EnvArgs(
            task_name=task_name,
            task_seed=89,
            max_steps=30,
            headless=True,  # 如果在自己电脑上跑可以设置成 true
            # timeout=15000,
        )

        # 特殊处理 openended 任务 可以自己操作 来处理human in the loop场景
        if task_name == "openended":
            AGENT_TEST.flags.enable_chat = True
            env_args.wait_for_user_message = True
            env_args.task_kwargs = {"start_url": "https://www.google.com"}

        # 将当前任务的实验参数添加到列表中
        exp_args_list.append(
            ExpArgs(
                agent_args=AGENT_TEST,
                env_args=env_args,
                logging_level=logging.INFO,
            )
        )

    for exp_args in tqdm(exp_args_list):
        benchmark = bgym.DEFAULT_BENCHMARKS[
            "webarena"
        ]() 
        # benchmark = bgym.DEFAULT_BENCHMARKS["assistantbench"]() 
        exp_args.agent_args.set_benchmark(
            benchmark, demo_mode=True
        )  # Override Some flags based on the benchmark.
        exp_args.agent_args.prepare()
        exp_args.prepare(exp_root=exp_dir)
        logging.info(f"Ready to run {exp_args}.")
        exp_args.run()
        logging.info("All jobs are finished. Calling agent_args.close() on all agents...")
        exp_args.agent_args.close()
        logging.info("Experiment finished.")
        # TODO: add ood result information to ExpResult
        # loading and printing results
        # exp_result = get_exp_result(exp_args.exp_dir)
        # exp_record = exp_result.get_exp_record()
        # for key, val in exp_record.items():
        #     print(f"{key}: {val}")


if __name__ == "__main__":
    main()
