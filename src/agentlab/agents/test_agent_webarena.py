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
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-11-20"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o4-mini-2025-04-16"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o1-2024-12-17"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["google/gemini-2.0-flash"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/o1-mini-2024-09-12"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4-1106-preview"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["local/Qwen2.5-7B-Instruct"],
    # chat_model_args=CHAT_MODEL_ARGS_DICT["openai/claude-3.7-sonnet"],
    flags=FLAGS_TEST,
    max_retry=3,
)


def main():
    exp_dir = "./test_results/"

    # 定义要测试的任务列表
    task_name_list = [  # 重新排序task_id后，每个Metacase后面标注是哪一个场景 哪一个metacase，scale是都否需要重新跑input，是否已经跑好
        ### shopping
        # "webarena.9", # ✅shopoing.unreachable.nonexist_column_report-disatisfaction-1 scale 4 个 可以用同一个input 已完成
        # "webarena.10", # ✅shopoing.unreachable.nonexist_column_draft-refund-message-1 scale 4 个 可以用同一个input 已完成
        # "webarena.164", # ✅shopping.misleading.unmacthed_instruction&url-1 n=1
        # "webarena.165", # ✅shopping.misleading.unmacthed_instruction&url-2 n=1
        # "webarena.166", # ✅shopping.misleading.unmacthed_instruction&url-3 n=1
        # "webarena.167", # ❌shopping.misleading.unmacthed_instruction&url-4 n=1 api_key 报错
        # "webarena.168", # ✅shopping.misleading.unmacthed_instruction&url-5 n=1
        ### shop_admin
        # "webarena.6", # ✅shop_admin.missinginfo.update-stock-1
        # "webarena.7", # ❌shop_admin.missinginfo.update-stock-2 报错说连接不上网页
        # "webarena.8", # ✅shop_admin.missinginfo.update-stock-3
        # "webarena.769", # ✅shop_admin.missinginfo.update-stock-4 n=4
        # "webarena.770", # ✅shop_admin.missinginfo.update-stock-5 n=4
        # "webarena.112", # ✅shop_admin.unreachable.search_in_nonexist_column-1 可以共用一个input (112-116 已修改scale input完成)
        # "webarena.453", # ✅shop_admin.unreachable.disable_nonexist_product-1 n=4 要求禁用满足A和B条件的商品，但是gpt4o把满足A或者满足B条件的商品都禁用了 后面修改了prompt要求只禁止一个
        # "webarena.454", # ✅shop_admin.unreachable.disable_nonexist_product-2 n=4
        # "webarena.455", # ✅shop_admin.unreachable.disable_nonexist_product-3 n=3
        # "webarena.456", # ✅shop_admin.unreachable.disable_nonexist_product-4 n=3
        # "webarena.457" # ✅shop_admin.unreachable.disable_nonexist_product-5 n=6 用作例子
        ## gitlab
        # "webarena.1", # ✅ webarena.gitlab.ambiguity.common_prefix_repo_name n=1
        # "webarena.481", # ✅ gitlab.unreachable.target-repo-name-misleading-1 n=8
        # "webarena.482", #  ❌gitlab.unreachable.target-repo-name-misleading-2 没有restart 导致仓库已存在
        # "webarena.483", # ✅gitlab.unreachable.target-repo-name-misleading-3 n=11
        # "webarena.484", # gitlab.unreachable.target-repo-name-misleading-4
        # "webarena.485", # ❌ gitlab.unreachable.target-repo-name-misleading-5 没有restart 导致仓库已存在
        # "webarena.486", # ❌gitlab.unreachable.target-repo-name-misleading-6 未能成功创建第二个仓库 点不动create new project
        # "webarena.487", # ❌ gitlab.TBD.repo-similiar-digits-letters-1 环境一直没有按照预期变化 可能是没有click上还是 可以考虑作为其他幻觉类型的input
        # "webarena.488", # ❌gitlab.TBD.repo-similiar-digits-letters-2  未能成功创建第二个仓库 点不动create new project
        # "webarena.489", # ✅gitlab.TBD.repo-similiar-digits-letters-3 n=8
        # "webarena.490", # ✅gitlab.TBD.repo-similiar-digits-letters-4 n=8用作repo-similiar-digits-letters（暂未幻觉） n=11用作unreachable.target-repo-name-misleading-7（已幻觉）
        # "webarena.491", # ✅gitlab.TBD.repo-similiar-digits-letters-5 n=8用作repo-similiar-digits-letters（暂未幻觉）n=10用作unreachable.target-repo-name-misleading-8
        # "webarena.799", # ❌gitlab.ambiguity.invite-member-common-name-1 n=19用作unexpected_transition.not-aware-failed-action 没有成功type 但模型误以为成功type了 检查模型是否意识到 但是邀请成员没有到达理想步骤
        # "webarena.800", # ❌gitlab.ambiguity.invite-member-common-name-2 n=16/13用作not-aware-failed-action 没有成功type 但模型误以为成功type了 检查模型是否意识到 但是邀请成员没有到达理想步骤
        # "webarena.801", # ✅gitlab.ambiguity.invite-member-common-name-3 n=11
        # "webarena.802", # ✅gitlab.ambiguity.invite-member-common-name-4 n=13用作unexpected_transition.not-aware-failed-action 报错了但是4o误以为已经成功邀请 n=15用作ambiguity.invite-member-common-name-4
        # "webarena.803", # ✅gitlab.ambiguity.invite-member-common-name-5 n=15
        # "webarena.804", # ❌gitlab.ambiguity.invite-member-common-name-6 n=15用作unexpected_transition.not-aware-failed-action 报错了但是4o误以为已经成功邀请
        # "webarena.567", # ✅gitlab.unreachable.not-my-repo-1 n=1
        # "webarena.568", # ✅gitlab.unreachable.not-my-repo-2 n=1看能否意识到unreachable
        ### reddit
        # "webarena.580", # ❌reddit.unreachable.create-forum-nonexist-column-1 执行完第一步后莫名停止
        # "webarena.581", # ✅reddit.unreachable.create-forum-nonexist-column-2 n=6用作human_in_loop.lack_of_neccessary_content-2
        # "webarena.582", # ✅reddit.unreachable.create-forum-nonexist-column-3 n=5用作unreachable n=6用作human_in_loop.lack_of_neccessary_content-3
        # "webarena.583", # ✅reddit.unreachable.create-forum-nonexist-column-4 n=5用作unreachable n=7用作human_in_loop.lack_of_neccessary_content-4
        # "webarena.584", # ✅eddit.unreachable.create-forum-nonexist-column-5 n=5用作unreachable  n=7用作human_in_loop.lack_of_neccessary_content-5 即创建forum还有一个必填项的内容没有给出
        ### map
        #"webarena.81", # ❌map.missinginfo.lack_means_of_transportation-1 都是api报错 n=0合适（在读题时就意识到ambiguity）还是n=后续搜索第二个地点到三个地点的时候看是否确定是walk还是car（无论选哪一个都是幻觉）
        #"webarena.82", # ❌map.missinginfo.lack_means_of_transportation-2
        #"webarena.83", # ❌map.missinginfo.lack_means_of_transportation-3
        #"webarena.84", # ❌map.missinginfo.lack_means_of_transportation-4
        #"webarena.85", # ❌map.missinginfo.lack_means_of_transportation-5

        #"webarena.74", # ✅ map.ambiguity.optimize-route-1 n=7 搜索2->3路线的click动作 但是应该n=0直接看模型是不是意识到optimal的ambiguity
        #"webarena.75", # ❌ map.ambiguity.optimize-route-2 api key 报错
        #"webarena.76", # ✅ map.ambiguity.optimize-route-3 n=7
        #"webarena.77", # ❌ map.ambiguity.optimize-route-4 n=7 但不是最理想unex
        #"webarena.78", # ❌ map.ambiguity.optimize-route-5 环境没有按照理想的改变

        #"webarena.32", #  ❌map.ambiguity.pick-best-hotel-1  都是api报错
        #"webarena.33", #  ❌map.ambiguity.pick-best-hotel-2
        #"webarena.34", #  ❌map.ambiguity.pick-best-hotel-3
        #"webarena.35", #  ❌map.ambiguity.pick-best-hotel-4
        #"webarena.36", #  ❌map.ambiguity.pick-best-hotel-5
        #"webarena.37", #  ❌map.ambiguity.pick-best-hotel-6
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
