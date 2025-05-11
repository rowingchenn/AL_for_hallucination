import os
import argparse
from pathlib import Path
import glob

def process_subfolder(subfolder_path: Path, dry_run: bool):
    """
    处理单个子文件夹：删除不需要的JSON文件并重命名目标文件。
    """
    subfolder_name = subfolder_path.name
    print(f"\nProcessing subfolder: {subfolder_path}")

    # 1. 识别特殊文件
    summary_file_name = "summary_info.json"
    # 找到所有符合 simulated_failed_input_FS*.json 模式的文件
    # Path.glob 返回的是生成器，我们将其转换为列表
    rename_candidate_paths = list(subfolder_path.glob("simulated_failed_input_FS*.json"))
    rename_candidate_names = [p.name for p in rename_candidate_paths]

    if not dry_run:
        print(f"  Found {len(rename_candidate_names)} rename candidates: {rename_candidate_names}")

    # 2. 删除其他 JSON 文件
    files_to_delete = []
    for item in subfolder_path.iterdir():
        if item.is_file() and item.name.endswith(".json"):
            if item.name == summary_file_name:
                if not dry_run: print(f"  Keeping summary file: {item.name}")
                continue
            if item.name in rename_candidate_names:
                if not dry_run: print(f"  Keeping rename candidate: {item.name}")
                continue
            
            # 如果运行到这里，说明这个JSON文件既不是summary也不是rename_candidate
            files_to_delete.append(item)

    if files_to_delete:
        print(f"  Files to delete in {subfolder_name}: {[f.name for f in files_to_delete]}")
        if not dry_run:
            for file_to_delete in files_to_delete:
                try:
                    file_to_delete.unlink()
                    print(f"    Deleted: {file_to_delete}")
                except OSError as e:
                    print(f"    Error deleting {file_to_delete}: {e}")
        else:
            print(f"  [DRY RUN] Would delete {len(files_to_delete)} JSON file(s).")
    elif not dry_run:
        print(f"  No other JSON files to delete in {subfolder_name}.")


    # 3. 重命名目标文件
    if len(rename_candidate_paths) == 1:
        old_path = rename_candidate_paths[0]
        new_name_str = f"{subfolder_name}.json"
        new_path = subfolder_path / new_name_str

        if old_path.name == new_name_str:
            print(f"  Skipping rename: File '{old_path.name}' is already correctly named.")
        elif new_path.exists():
            # 检查是否是同一个文件（理论上不太可能，因为上面已经排除了）
            # 但如果 new_path 存在且不是 old_path，则有冲突
            try:
                if old_path.samefile(new_path): # new_path 是 old_path 的硬链接或自身
                     print(f"  Skipping rename: File '{old_path.name}' and target '{new_name_str}' are the same file.")
                else:
                     print(f"  Skipping rename: Target file '{new_name_str}' already exists in '{subfolder_name}' and is a different file.")
            except FileNotFoundError: # old_path 可能在 dry run 的想象中被删了，或者 new_path 存在但 old_path 不存在
                 print(f"  Skipping rename: Target file '{new_name_str}' already exists in '{subfolder_name}'. Cannot verify if it's the same as '{old_path.name}'.")

        else:
            action_prefix = "[DRY RUN] Would rename" if dry_run else "Renaming"
            print(f"  {action_prefix} '{old_path.name}' to '{new_name_str}'")
            if not dry_run:
                try:
                    old_path.rename(new_path)
                    print(f"    Successfully renamed.")
                except OSError as e:
                    print(f"    Error renaming {old_path.name} to {new_name_str}: {e}")

    elif len(rename_candidate_paths) == 0:
        print(f"  No 'simulated_failed_input_FS*.json' file found in '{subfolder_name}' to rename.")
    else: # len(rename_candidate_paths) > 1
        print(f"  Warning: Multiple 'simulated_failed_input_FS*.json' files found in '{subfolder_name}'.")
        print(f"           Files: {[p.name for p in rename_candidate_paths]}")
        print(f"           Skipping rename to '{subfolder_name}.json' to avoid overwriting.")


def main():
    parser = argparse.ArgumentParser(
        description="Deletes specified JSON files from subfolders and renames one target JSON file."
    )
    parser.add_argument(
        "parent_directory",
        type=str,
        help="The parent directory containing subfolders to process."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done, without actually deleting or renaming files."
    )
    args = parser.parse_args()

    parent_dir = Path(args.parent_directory)
    if not parent_dir.is_dir():
        print(f"Error: Provided path '{args.parent_directory}' is not a directory or does not exist.")
        return

    print(f"Starting processing for parent directory: {parent_dir.resolve()}")
    if args.dry_run:
        print("--- DRY RUN MODE ENABLED --- (No actual changes will be made)")

    subfolder_count = 0
    for item in parent_dir.iterdir():
        if item.is_dir():
            subfolder_count += 1
            process_subfolder(item, args.dry_run)
    
    if subfolder_count == 0:
        print("No subfolders found in the specified parent directory.")
    else:
        print(f"\nProcessed {subfolder_count} subfolder(s).")
    
    if args.dry_run:
        print("--- DRY RUN MODE FINISHED ---")

if __name__ == "__main__":
    main()