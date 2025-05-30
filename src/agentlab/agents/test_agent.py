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


FLAGS_TEST = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=False,
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
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o4-mini-2025-04-16"],
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
    # task_name_list = [
    # # "workarena.servicenow.infeasible-navigate-and-create-user-with-reason-l2",
    # # "workarena.servicenow.infeasible-navigate-and-order-apple-mac-book-pro15-l2",
    # # "workarena.servicenow.workload-balancing-small-l2",
    # # "workarena.servicenow.order-standard-laptop",
    # # "webarena.2",
    # # "assistantbench.validation.1",
    # # "workarena.servicenow.infeasible-navigate-and-create-user-l2",
    # # "workarena.servicenow.infeasible-navigate-and-create-incident-l2",
    # # "workarena.servicenow.infeasible-navigate-and-create-change-request-l2",
    # # "workarena.servicenow.infeasible-navigate-and-create-problem-l2",
    # # "workarena.servicenow.infeasible-navigate-and-create-hardware-asset-l2",
    # # "workarena.servicenow.infeasible-navigate-and-order-standard-laptop-l2",
    # # "workarena.servicenow.infeasible-navigate-and-order-sales-laptop-l2",
    # # "workarena.servicenow.infeasible-navigate-and-order-developer-laptop-l2",
    # # "workarena.servicenow.infeasible-navigate-and-order-ipad-pro-l2",
    # # "workarena.servicenow.infeasible-navigate-and-order-ipad-mini-l2",
    # # "workarena.servicenow.infeasible-navigate-and-order-apple-watch-l2",
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
    # ]

    task_name_list = [
        # "workarena.servicenow.basic-filter-problems-and-mark-duplicates-small-l2",
        # "workarena.servicenow.priority-filter-problems-and-mark-duplicates-small-l2",
        # "workarena.servicenow.high-priority-filter-problems-and-mark-duplicates-small-l2",
        # "workarena.servicenow.workload-balancing-small-l2",
        # "workarena.servicenow.work-assignment-small-l2",
        # "workarena.servicenow.priority-assignment-small-l2",
        # "workarena.servicenow.two-changes-basic-uniform-risk-change-request-scheduling-l2",
        # "workarena.servicenow.two-changes-fix-basic-uniform-risk-change-request-scheduling-l2",
        # "workarena.servicenow.three-changes-basic-uniform-risk-change-request-scheduling-l2",
        # "workarena.servicenow.two-changes-priority-uniform-risk-change-request-scheduling-l2",
        # "workarena.servicenow.two-changes-wide-priority-varied-risk-change-request-scheduling-l2",
        # "workarena.servicenow.two-changes-tight-priority-varied-risk-change-request-scheduling-l2",
        # "workarena.servicenow.three-changes-priority-uniform-risk-change-request-scheduling-l2",
        # "workarena.servicenow.dashboard-retrieve-catalog-and-max-order-developer-laptop-l2",
        # "workarena.servicenow.dashboard-retrieve-incident-and-min-create-incident-l2",
        # "workarena.servicenow.dashboard-retrieve-incident-and-max-request-apple-watch-l2",
        # "workarena.servicenow.get-warranty-expiration-date-l2",
        # "workarena.servicenow.filter-requested-items-and-order-developer-laptop-l2",
        # "workarena.servicenow.basic-expense-management-small-l2",
        # "workarena.servicenow.filter-random-expenses-and-find-total-return-small-l2",
        # "workarena.servicenow.filter-three-items-uniform-expenses-and-find-total-return-large-l2",
        # "workarena.servicenow.dashboard-retrieve-catalog-and-mean-order-developer-laptop-l2",
        # "workarena.servicenow.dashboard-retrieve-incident-and-mean-create-incident-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-user-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-incident-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-change-request-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-problem-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-hardware-asset-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-user-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-incident-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-change-request-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-problem-l2",
        # "workarena.servicenow.infeasible-navigate-and-create-hardware-asset-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-standard-laptop-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-sales-laptop-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-developer-laptop-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-ipad-pro-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-ipad-mini-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-apple-watch-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-apple-mac-book-pro15-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-development-laptop-p-c-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-loaner-laptop-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-standard-laptop-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-sales-laptop-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-developer-laptop-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-ipad-pro-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-ipad-mini-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-apple-watch-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-apple-mac-book-pro15-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-development-laptop-p-c-l2",
        # "workarena.servicenow.infeasible-navigate-and-order-loaner-laptop-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-asset-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-user-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-incident-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-change-request-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-hardware-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-service-catalog-item-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-asset-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-user-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-incident-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-change-request-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-hardware-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-filter-service-catalog-item-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-asset-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-user-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-incident-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-change-request-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-hardware-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-service-catalog-item-list-with-reason-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-asset-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-user-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-incident-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-change-request-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-hardware-list-l2",
        # "workarena.servicenow.infeasible-navigate-and-sort-service-catalog-item-list-l2",
        "webarena.655",
        "webarena.656",
        "webarena.657",
        "webarena.689",
        "webarena.690",
        "webarena.691",
        "webarena.692",
        "webarena.693",
        "webarena.792",
        "webarena.793",
        "webarena.794",
        "webarena.795",
        "webarena.796",
        "webarena.797",
        "webarena.798",
    ]

    # 初始化空的实验参数列表
    exp_args_list = []

    # 循环生成每个任务的实验参数
    for task_name in task_name_list:
        env_args = EnvArgs(
            task_name=task_name,
            task_seed=9,
            max_steps=30,
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
            "workarena_l2_agent_curriculum_eval"
            # "webarena"
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
