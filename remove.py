import csv
import os
import shutil

# 核心配置参数（可根据实际情况调整）
CSV_FILE_PATH = "/disk4/lyz/FlexiMMT/benchmark_new/captions_inf/val_images.csv"
# 路径的根目录（拼接相对路径为绝对路径）
ROOT_DIR = "/disk4/lyz/FlexiMMT/"
# 是否覆盖已存在的目标文件夹（建议先设为False，确认无误后再改为True）
OVERWRITE = False

def move_mask_folders_with_action():
    # 检查CSV文件是否存在
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ 错误：CSV文件不存在 -> {CSV_FILE_PATH}")
        return
    
    # 初始化统计变量
    success_count = 0  # 移动成功
    fail_count = 0     # 移动失败
    skip_count = 0     # 跳过（文件已存在/空行等）

    # 读取CSV文件并处理每一行
    with open(CSV_FILE_PATH, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # 遍历每一行（如果CSV有表头，取消下面注释跳过表头）
        # next(csv_reader)
        
        for row_num, row in enumerate(csv_reader, 1):
            # 跳过空行
            if not row:
                print(f"📌 第{row_num}行：空行，跳过")
                skip_count += 1
                continue
            
            # 检查是否有至少两列数据
            if len(row) < 2:
                print(f"❌ 第{row_num}行：列数不足（需要至少2列），跳过")
                fail_count += 1
                continue
            
            # 提取第一列（图片路径）和第二列（动作），并去除首尾空格
            img_relative_path = row[0].strip()
            action_str = row[1].strip()
            
            # 检查列值是否为空
            if not img_relative_path:
                print(f"📌 第{row_num}行：第一列（图片路径）为空，跳过")
                skip_count += 1
                continue
            if not action_str:
                print(f"📌 第{row_num}行：第二列（动作）为空，跳过")
                skip_count += 1
                continue

            # --------------------------
            # 步骤1：构造原mask文件夹的路径
            # --------------------------
            # 1. 将 target_images 替换为 target_masks
            mask_base_relative = img_relative_path.replace("target_images", "target_masks")
            # 2. 拼接动作字符串（格式：图片路径+动作）
            original_mask_folder_relative = f"{mask_base_relative}+{action_str}"
            # 3. 拼接为绝对路径
            original_mask_folder = os.path.join(ROOT_DIR, original_mask_folder_relative)

            # 检查原mask文件夹是否存在且是有效目录
            if not os.path.isdir(original_mask_folder):
                print(f"❌ 第{row_num}行：原mask文件夹不存在 -> {original_mask_folder}")
                fail_count += 1
                continue

            # --------------------------
            # 步骤2：构造新mask文件夹的路径
            # --------------------------
            # 分割路径，找到 target_masks 位置，提取最终文件夹名，拼接新路径
            path_parts = original_mask_folder_relative.split('/')
            try:
                # 找到 target_masks 的索引
                target_masks_idx = path_parts.index("target_masks")
                # 提取最后一部分（完整的mask文件夹名，如 animal_bird1+animal_wolf1.png+kangaroo+human2animal_1）
                mask_folder_full_name = path_parts[-1]
                # 拼接新的相对路径：benchmark_new/target_masks/[完整文件夹名]
                new_mask_relative = os.path.join(*path_parts[:target_masks_idx+1], mask_folder_full_name)
                # 拼接为绝对路径
                new_mask_folder = os.path.join(ROOT_DIR, new_mask_relative)
            except ValueError:
                print(f"❌ 第{row_num}行：路径不包含target_masks -> {original_mask_folder_relative}")
                fail_count += 1
                continue

            # --------------------------
            # 步骤3：处理文件夹移动
            # --------------------------
            # 检查目标文件夹是否已存在
            if os.path.exists(new_mask_folder):
                if OVERWRITE:
                    # 覆盖：先删除已存在的目标文件夹（谨慎使用！）
                    print(f"⚠️  第{row_num}行：目标文件夹已存在，删除后覆盖 -> {new_mask_folder}")
                    shutil.rmtree(new_mask_folder)
                else:
                    print(f"📌 第{row_num}行：目标文件夹已存在，跳过（如需覆盖请设置OVERWRITE=True） -> {new_mask_folder}")
                    skip_count += 1
                    continue

            # 确保目标目录的父文件夹存在（防止创建失败）
            os.makedirs(os.path.dirname(new_mask_folder), exist_ok=True)

            # 移动文件夹
            try:
                shutil.move(original_mask_folder, new_mask_folder)
                print(f"✅ 第{row_num}行：移动成功 -> {original_mask_folder} -> {new_mask_folder}")
                success_count += 1
            except Exception as e:
                print(f"❌ 第{row_num}行：移动失败 -> {original_mask_folder}，错误：{str(e)}")
                fail_count += 1

    # 输出最终统计结果
    print("\n==================== 处理完成 ====================")
    print(f"✅ 成功移动：{success_count} 个文件夹")
    print(f"❌ 移动失败：{fail_count} 个文件夹")
    print(f"📌 跳过处理：{skip_count} 个文件夹")

if __name__ == "__main__":
    move_mask_folders_with_action()