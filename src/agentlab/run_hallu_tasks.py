import sys
from pathlib import Path

# ✅ 添加 src/ 到 PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import logging
from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

# ✅ 你刚才定义的自定义 benchmark（会 monkey patch）
from agentlab.hallu_tasks.custom_tasks import MyWebArenaBenchmark

# ✅ Study 构造器
from agentlab.experiments.study import make_study

# ✅ 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ✅ 日志设置
logging.basicConfig(level=logging.INFO)
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

# ✅ Agent 配置（你之前已经写好了）
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
        action_set=dp.bgym.HighLevelActionSetArgs(
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
    max_prompt_tokens=128000,
    extra_instructions=(
        "If you meet some problems and you can't solve, "
        "please don't do too many meaningless retries. Just report the problem to the user."
    ),
)

AGENT_TEST = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["google/gemini-2.0-flash"],
    flags=FLAGS_TEST,
    max_retry=3,
)

# ✅ 主逻辑入口
def main():
    logger.info("Creating study with custom WebArena tasks...")

    # ✅ 会加载你的 JSON、patch 掉 test.raw.json
    benchmark = MyWebArenaBenchmark()

    study = make_study(
        agent_args=AGENT_TEST,
        benchmark=benchmark,
        suffix="custom_webarena_hallu",  # 输出文件夹名字
    )

    logger.info("Starting study...")
    study.run(n_jobs=1)
    logger.info("Study finished.")

if __name__ == "__main__":
    main()
