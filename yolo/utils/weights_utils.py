#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :model_utils.py
# @Time      :2025/6/13 11:24:24
# @Author    :雨霓同学
# @Project   :SafeH
# @Function  :用于将训练好的模型拷贝到模型存储目录

from datetime import datetime
from pathlib import Path
import shutil
from .paths import CHECKPOINTS_DIR
import os
import re

def copy_checkpoint_models(train_dir, model_filename, logger=None):
    """
    复制训练好的模型到指定的目录下
    :param train_dir: 训练的模型存放地址 (e.g., Path('runs/train/exp'))
    :param model_filename: 模型名称，假设其不带 .pt 的部分就是 pretrained_name (e.g., 'yolov5s.pt')
    :param logger: 日志记录器实例
    :return: 不需要返回值
    """
    if not train_dir or not isinstance(train_dir, Path):
        if logger:
            logger.error("无效的训练目录，跳过模型复制")
        return

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 提取 pretrained_name，假设 model_filename 的不带 .pt 的部分即为 pretrained_name
    pretrained_name = model_filename.replace(".pt", "")

    # 从 train_dir.name 中提取 train_id
    # 假设 train_dir.name 的格式是 "train{id}" 或 "exp{id}"
    match = re.search(r'(train|exp)(\d+)', train_dir.name)
    if match:
        train_id = f"{match.group(1)}{match.group(2)}"  # 组合成 train1, exp2 这样的格式
    else:
        # 如果 train_dir.name 不符合预期格式，可以使用完整的 train_dir.name 作为标识
        train_id = train_dir.name
        if logger:
            logger.warning(f"无法从训练目录名 '{train_dir.name}' 中解析出 train_id，将使用完整目录名作为ID。")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if not os.access(CHECKPOINTS_DIR, os.W_OK):
        if logger:
            logger.info(f"检查目录：{CHECKPOINTS_DIR}不可写，跳过模型复制")
        raise OSError("检查目录不可写")

    for model_type in ["best", "last"]:
        src_path = train_dir / "weights" / f"{model_type}.pt"
        if src_path.exists():
            # 构建新的 checkpoint_name
            # 格式：{train{id}}-{timestamp}-{pretrained_name}-{best/last}.pt
            checkpoint_name = f"{train_id}-{date_str}-{pretrained_name}-{model_type}.pt"
            dest_path = CHECKPOINTS_DIR / checkpoint_name
            shutil.copy(src_path, dest_path)
            if logger:
                logger.info(f"复制模型文件：{model_type}.pt -> {dest_path}")
        else:
            if logger:
                logger.info(f"模型文件 {model_type}.pt 不存在：{src_path}")