import os
import json

# 目标文件夹路径
target_dir = "."

# 输出 JSON 文件名
output_file = "truncate_map_unachievable_original_webarena.json"

# 列出目标目录下的所有子文件夹（忽略文件）
all_folders = [f for f in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, f))]

# 排序（按文件夹名排序，可以改成按创建时间等）
all_folders.sort()

# 存储新名字
new_folder_names = []

# 遍历并重命名
for idx, old_name in enumerate(all_folders, start=1):
    new_name = old_name

    # 去掉末尾的 "_89"（如果有）
    if new_name.endswith("_89"):
        new_name = new_name[:-3]

    # 加上后缀
    new_name = f"{new_name}.unachievable_original_webarena_tasks-{idx}"

    # 重命名文件夹
    os.rename(
        os.path.join(target_dir, old_name),
        os.path.join(target_dir, new_name)
    )

    # 保存新的名字
    new_folder_names.append(new_name)

# 把所有新名字保存成 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_folder_names, f, ensure_ascii=False, indent=2)
    f.write("\n\n")  # 写一个额外空行

print(f"重命名完成，共处理 {len(new_folder_names)} 个文件夹，结果保存到 {output_file}")
