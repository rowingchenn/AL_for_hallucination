import bgym
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
import os

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
        long_description=False,
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
    extra_instructions="If you meet some problems and you can't solve, please don't do to many meaningless retries. Just report the problem to the user.",
    # extra_instructions=None,
)

AGENT_TEST = GenericAgentArgs(
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o3-mini-2025-01-31"],
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-11-20"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4-1106-preview"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["local/Qwen2.5-7B-Instruct"],
    flags=FLAGS_TEST,
    max_retry=3,
)


def main():
    exp_dir = "./test_results/"

    env_args = EnvArgs(
        # task_name="webarena.692",
        # task_name="workarena.servicenow.infeasible-navigate-and-order-apple-mac-book-pro15-l2",  # L2 is multi-tab
        # task_name="workarena.servicenow.workload-balancing-small-l2",
        # task_name="workarena.servicenow.order-standard-laptop",
        # task_name="assistantbench.validation.1",
        task_name="workarena.servicenow.infeasible-navigate-and-create-user-l2",
        task_seed=89,
        max_steps=20,
        headless=False,
        # timeout=15000,
    )

    if env_args.task_name == "openended":
        AGENT_TEST.flags.enable_chat = True
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": "https://www.google.com"}

    exp_args_list = [
        ExpArgs(
            agent_args=AGENT_TEST,
            env_args=env_args,
            logging_level=logging.INFO,
        ),
    ]

    for exp_args in exp_args_list:
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
