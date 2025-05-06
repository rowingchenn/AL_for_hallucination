import os
import shutil

# 源目录：实验目录们
SOURCE_ROOT = "/Users/pujiayue/Documents/research/llm agent hallucination/AL_for_hallucination/hallucination_traj_sample/Webarena_original_infeasible"

# 目标目录：按风险类型分类保存 json 文件
DEST_ROOT = "/Users/pujiayue/Documents/research/llm agent hallucination/llm_agent_hallucination_data/dataset_all/Webarena"

# 风险类型关键词列表（已加入 misleading）
RISK_TYPES = [
    "ambiguity", "human_in_loop", "missinginfo",
    "unreachable", "misleading", "unachievable_original"
]

def find_risk_type(filename):
    for risk in RISK_TYPES:
        if risk in filename:
            return risk
    return None

def main():
    print(f"📂 正在从实验目录读取: {SOURCE_ROOT}")
    print(f"📦 目标输出目录: {DEST_ROOT}")

    for folder in os.listdir(SOURCE_ROOT):
        folder_path = os.path.join(SOURCE_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"\n🔍 处理实验目录: {folder_path}")
        # 找到该目录下唯一一个非 summary_info.json 的 .json 文件
        json_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".json") and f != "summary_info.json"
        ]

        if not json_files:
            print("⚠️ 未找到符合条件的 .json 文件")
            continue

        if len(json_files) > 1:
            print(f"⚠️ 警告：找到多个 JSON 文件，仅处理第一个: {json_files[0]}")

        json_file = json_files[0]
        risk_type = find_risk_type(json_file)

        if not risk_type:
            print(f"⚠️ 无法识别风险类型：{json_file}")
            continue

        # 构造移动路径
        src_path = os.path.join(folder_path, json_file)
        target_dir = os.path.join(DEST_ROOT, risk_type)
        os.makedirs(target_dir, exist_ok=True)
        dst_path = os.path.join(target_dir, json_file)

        # ✅ 若目标文件已存在则跳过
        if os.path.exists(dst_path):
            print(f"⏭️ 已存在，跳过: {dst_path}")
            continue

        shutil.move(src_path, dst_path)
        print(f"✅ 已移动 {json_file} → {dst_path}")

if __name__ == "__main__":
    main()
