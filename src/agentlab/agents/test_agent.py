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
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o3-mini-2025-01-31"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-11-20"],
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
        # "workarena.servicenow.infeasible-navigate-and-create-user-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-apple-mac-book-pro15-l2",
        # "workarena.servicenow.workload-balancing-small-l2",
        # "workarena.servicenow.order-standard-laptop",
        # "webarena.2",
        # "assistantbench.validation.1",
        # "workarena.servicenow.infeasible-navigate-and-create-user-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-incident-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-change-request-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-problem-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-hardware-asset-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-standard-laptop-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-sales-laptop-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-developer-laptop-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-ipad-pro-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-ipad-mini-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-apple-watch-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-apple-mac-book-pro15-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-development-laptop-p-c-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-loaner-laptop-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-asset-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-user-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-incident-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-change-request-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-hardware-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-service-catalog-item-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-asset-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-user-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-incident-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-change-request-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-hardware-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-service-catalog-item-list-l2",
        # "webarena.2",
        # "webarena.3",
        # "webarena.5",
        # "webarena.6",
        # "webarena.7",
        #"webarena.11",
        #"webarena.16",
        "webarena.6.5",
        # "webarena.17",
    ]

    # 初始化空的实验参数列表
    exp_args_list = []

    # 循环生成每个任务的实验参数
    for task_name in task_name_list:
        env_args = EnvArgs(
            task_name=task_name,
            task_seed=89,
            max_steps=20,
            headless=False,  # 如果在自己电脑上跑可以设置成 true
            # timeout=15000,
        )

        # 特殊处理 openended 任务
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
            # "workarena_l2_agent_curriculum_eval"
            "webarena"
        ]()  # 如果跑 WebArena 的 benchmark 需要换成 bgym.DEFAULT_BENCHMARKS["webarena"]()
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
