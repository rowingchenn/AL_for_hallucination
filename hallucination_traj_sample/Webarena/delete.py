import os
from pathlib import Path

def delete_non_summary_json_files(directory):
    """
    递归删除指定目录及其所有子目录中的非summary*.json和非truncate*.json的json文件

    Args:
        directory (str): 要处理的目录路径

    Returns:
        int: 删除的文件数量
    """
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") and not (file.startswith("summary") or file.startswith("truncate")):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")

    return count

if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent

    # 删除非summary*.json和非truncate*.json的json文件
    deleted_count = delete_non_summary_json_files(current_dir)

    print(f"\n总共删除了 {deleted_count} 个符合条件的json文件")
