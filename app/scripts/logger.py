# coding:utf-8
# @FileName: logger.py
# @Author: BLC
# @Time: 2025/6/20 00:02
# @Project: SafeH
# @Function:
import logging
import os
from datetime import datetime

def setup_logger(log_dir, encoding="utf-8"):
    """
    设置日志记录器
    :param log_dir: 日志文件保存的目录
    :return: 配置好的日志记录器
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建日志文件名称
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 配置日志记录器
    logger = logging.getLogger("SafeH")
    logger.setLevel(logging.INFO)

    # 避免重复添加处理器
    if not logger.handlers:  # 处理器包括文件处理器和控制台处理器，分别将日志写入指定文件路径和控制台
        file_handler = logging.FileHandler(log_file, encoding=encoding)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"日志文件：{log_file}")
    logger.info(f"日志编码：{encoding}")
    logger.info(f"日志级别：INFO")
    logger.info("日志初始化完成")

    return logger