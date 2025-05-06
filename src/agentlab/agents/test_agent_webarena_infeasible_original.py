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
os.environ["WA_REDDIT"] = "http://111.229.174.217:18080"
os.environ["WA_GITLAB"] = "http://111.229.174.217:19001"
os.environ["WA_WIKIPEDIA"] = "http://111.229.174.217:19003"
os.environ["WA_MAP"] = "http://111.229.174.217:1443"
os.environ["WA_HOMEPAGE"] = "http://111.229.174.217:18081"
# os.environ["WA_FULL_RESET"] = "http://111.229.174.217:17565" # /satus 访问不了，因此暂时跳过
os.environ["WA_FULL_RESET"] = ""

FLAGS_TEST = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=True,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=False,
        extract_clickable_tag=True,
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
    #chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-11-20"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o1-2024-12-17"],
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o4-mini-2025-04-16"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["google/gemini-2.0-flash"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o1-mini-2024-09-12"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4-1106-preview"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["local/Qwen2.5-7B-Instruct"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/anthropic/claude-3.7-sonnet"],
    flags=FLAGS_TEST,
    max_retry=3,
)


def main():
    exp_dir = "./test_results_webarena_original_infeasible/"

    # 定义要测试的任务列表
    task_name_list = [
        #"webarena.22",
        #"webarena.24",
        #"webarena.101",
        #"webarena.115",
        #"webarena.166",
        #"webarena.168",
        #"webarena.183", # ❌ # o1没幻觉，但还是不行 待定 感觉task不好
        #"webarena.191",
        #"webarena.201",
        #"webarena.218", # ❌ 不够好 
        #"webarena.219", # ❌ 不够好 # n = 11 ✅
        # "webarena.225", # repetitive_action-9 n=5
        # "webarena.234", # repetitive_action-10 n=14
        #"webarena.235", # ❌ o1❌
        # "webarena.247",
        #"webarena.253", # ❓ 4o n=6 还是 o4-mini-task16 n=13
        #"webarena.301",
        #"webarena.302", # repetitive_action-11 n=19; repetitive_action_easier-11 n=23都是一直在navigate page1和2 不做出判断
        #"webarena.313",
        #"webarena.368",
        #"webarena.376", # o1 n=6
        #"webarena.491", # 修改了task instruction  Click on the 'Orders' link under the SALES menu to navigate to the orders page then click on the 'Filters' button to open the filter options. Then move on to try to complete the task.
        #"webarena.723", #❌ # 虽然我感觉挺好了但是看model的thinking感觉还是不够好 # 修改 task instruction Type 'FTorrez81' into the search box to find submissions by this user.
        #"webarena.726", # ❌ repetitive_action-12 n=25 repetitive_action_easier-12 n=27 要求只搜索人名
        # "webarena.783", # 需要修改 n=8
        #"webarena.789", # 可以是underspecified_missinginfo(没有给出用户名和密码) n=7; 也可以是repetitive_action-13 n=13
        #"webarena.790", # n = 5
        #"webarena.791", # ❌了 30 步满了还没做出判断 需要重启
        #"webarena.792", # ❌30 步满了还不对
        #"webarena.793", # ❌30 步满了还不对 可能需要在task里写明正确步骤
        #"webarena.794", # n = 3✅ 就可以判断不可以change address after the order is placed
        #"webarena.795", # repetitive_action-14， n = 3 ✅
        #"webarena.796", # ✅n = 2 就可以判断不可以change address after the order is placed
        #"webarena.797", # ✅n = 2 就可以判断不可以change address after the order is placed 但是最好可以scroll一下
        # "webarena.798", 
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
            # timeout=30000,
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
        benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l2_agent_curriculum_eval"]()
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
