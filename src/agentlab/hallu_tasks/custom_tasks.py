# agentlab/hallu_tasks/custom_tasks.py

import importlib.resources
import json
from pathlib import Path
from browsergym.experiments import EnvArgs
from bgym import Benchmark, HighLevelActionSetArgs

import webarena  # 重要：WebArena 模块被 monkey patch

# 🔧 Monkey Patch：覆盖 test.raw.json 加载逻辑
def patch_webarena_config(custom_json_path: Path):
    """
    Monkey patch WebArena so that test.raw.json is read from our own JSON file.
    """
    import importlib.resources

    class FakeFile:
        def __init__(self, path):
            self.path = path

        def joinpath(self, name):
            return self  # mock joinpath chaining

        def read_text(self, encoding=None):
            return self.path.read_text(encoding="utf-8")

    import webarena

    # monkey patch importlib.resources.files(webarena)
    importlib.resources.files = lambda package: FakeFile(custom_json_path)


# ✅ 自定义 benchmark（任务加载方式和官方一样）
class MyWebArenaBenchmark(Benchmark):
    def __init__(self):
        custom_path = Path(__file__).parent / "webarena_hallu_tasks.json"
        patch_webarena_config(custom_path)

        # 任务名按惯例用 webarena.<task_id>
        env_args_list = [
            EnvArgs(
                task_name=f"webarena.{task['task_id']}",
                task_seed=42,
                max_steps=40,
                headless=True,
            )
            for task in json.loads(custom_path.read_text())
        ]

        super().__init__(
            name="my_custom_webarena",
            high_level_action_set_args=HighLevelActionSetArgs(subsets=["bid"]),
            is_multi_tab=True,
            supports_parallel_seeds=False,
            env_args_list=env_args_list,
            backends=["webarena"],
        )
