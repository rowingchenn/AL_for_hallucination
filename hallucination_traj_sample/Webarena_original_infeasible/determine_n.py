import os
import json

# 目标文件夹路径
target_dir = "."

# 输出 JSON 文件
output_file = "truncate_map_unachievable_original_webarena_final.json"

# 保存结果
truncate_map = {}

# 遍历所有子文件夹
for subdir in os.listdir(target_dir):
    subdir_path = os.path.join(target_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue

    log_path = os.path.join(subdir_path, "experiment.log")
    if not os.path.isfile(log_path):
        print(f"警告: {subdir} 中没有 experiment.log 文件，跳过。")
        continue

    # 读取 experiment.log
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    action_count = 0  # 当前是第几个 action
    found = False     # 是否找到 send_msg_to_user
    total_actions = 0  # action 总数

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "action:":
            total_actions += 1
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if "send_msg_to_user" in next_line:
                    truncate_map[subdir] = action_count
                    found = True
                    break
            action_count += 1
            i += 2  # 跳过动作和动作内容
        else:
            i += 1

    if not found:
        print(f"提示: {subdir} 中没有找到 send_msg_to_user，使用最后一个 action 编号 {total_actions - 1}") # 没有找到就把 n 定为最后一个action
        truncate_map[subdir] = max(total_actions - 1, 0)  # 防止 -1，至少是0

# 写入 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(truncate_map, f, ensure_ascii=False, indent=2)

print(f"处理完成，共 {len(truncate_map)} 个子目录，保存到 {output_file}")
