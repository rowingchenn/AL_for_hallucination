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
    #chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini"],
    #chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-mini-2024-07-18"],
    #chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o3-mini-2025-01-31"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-11-20"],
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
    exp_dir = "./test_results_webarena_unexpected_transition/"

    # 定义要测试的任务列表
    task_name_list = [
        #"webarena.4",
        #"webarena.13",
        #"webarena.42",
        #"webarena.95",
        #"webarena.208",
        #"webarena.288",
        #"webarena.470",
        #"webarena.32", # 16-18的第一步后api_base就变回官网了，不知道咋改
        "webarena.86", # 223,218的第一步后api_base就变回官网了，不知道咋改
        #"webarena.248",
        #"webarena.363",
        #"webarena.380",
        #"webarena.144", # 22的第一步后api_base就变回官网了，不知道咋改
        #"webarena.329", # 50的第一步后api_base就变回官网了，不知道咋改
        #"webarena.229",
        #"webarena.284",
        #"webarena.324", 
        #"webarena.506", # timeout，已解决
        #"webarena.692",
        #"webarena.600",
        #"webarena.609",
        #"webarena.714",
        #"webarena.732",
        #"webarena.293",
        #"webarena.390", # 网站加载超时
        #"webarena.411",
        #"webarena.480",
        #"webarena.808" 

        # 飞书下面workArena下面的 unexpected transition，在正常的task上
        # 1.error, 环境会报错，能不能在下一步action意识到上一步又报错了，而是觉得上一步action已经完成了
        # 2.accident，正常跑，人为把这一步的observation捏造成上一步的
        # 拿o4-mini挑20个webarena的task（每一个环境有五个），标注出哪一个step是关键的
        # shopping_admin: 4, 13(可以修改为填写不存在的栏的task)，42，95，208,288,470(按cancel之后还是同样的界面，是否直接认为是cancel掉了，781)
        # map: 18-》32（在选择car并且按了go之后不动环境，搜索得到的结果还是car/foot？）, 223-》218-》81（click go之后环境不动，是否认为是不存在搜索结果还是认为可能是没有click上go），248（同223），363（同223），380
        # shopping:22（Click on the 'Page 2' link）-》144（Click on the 'View All' link 后环境不变），50->329（Click on the 'Page 3' link），229（sorted by），284（sort by），324/325（sort by），506（Click on the 'Next' button后不动，是否会认为已经买完了），692(type完email和name后保持环境不变，会不会认为已经填写完了)
        # reddit: 600-604( Type xxx into the 'Body' textbox. 其实没type进去)，609-612（Type 'xx' into the 'Title' textbox.这是必填项，但是没填进去），714-716（sort & thumbs down 两个都可以设置hallu），732
        # gitlab：293-297（Click on the 'Next' link翻页，但在中间一步没有翻页成功），390（没有成功填入），411-414（需要ensure是不是在main branch），480-485（invite之后环境不变），808（ Click on the 'Create merge request' button后环境不动）
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
            #timeout=30000,
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
