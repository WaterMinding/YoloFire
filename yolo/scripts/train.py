#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :yolo_train.py
# @Time     :2025/6/12 15:25:42
# @Author   :雨霓同学
# @Project  :SafeH
# @Function :训练脚本入口, 集成utils模块
import sys
from pathlib import Path

# 将 yolo/ 目录添加到 sys.path 中
yolo_path = Path(__file__).parent.parents[0]
if not str(yolo_path) in sys.path:
    sys.path.insert(0, str(yolo_path))

import argparse
import logging
from ultralytics import YOLO

from utils import (
    setup_logger,
    load_yaml,
    merge_args,
    LOGGING_DIR,
    PRETRAINED_DIR,
    log_dict, 
    rename_logfile,
    get_device_info,
    get_args_info,
    get_result_info,
    get_dataset_info,
    copy_checkpoint_models  # 实现模型的拷贝
)



def parse_args():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Training")
    parser.add_argument('--data', type=str, default="data.yaml", help="YAML配置文件路径")
    parser.add_argument('--batch', type=int, default=32, help="批量大小")
    parser.add_argument('--epochs', type=int, default=100, help="训练轮数")
    parser.add_argument('--imgsz', type=int, default=640, help="图像大小")
    parser.add_argument('--device', type=str, default="0", help="设备ID")
    parser.add_argument('--workers', type=int, default=16, help="工作线程数")

    # 项目自定义的参数
    parser.add_argument('--weights', type=str, default="yolov8n.pt", help="预训练权重路径")
    parser.add_argument('--log_encoding', type=str, default="utf-8", help="日志编码格式")
    parser.add_argument('--log_level', type=str, default="INFO", help="日志级别")
    parser.add_argument('--use_yaml', type=bool, default=True, help="是否使用YAML配置文件")

    return parser.parse_args()


def run_training(model, yolo_args):
    """
    对训练过程做个简单的封装
    """
    results = model.train(**vars(yolo_args))
    return results


def main():
    args = parse_args()
    logger = logging.getLogger("YOLO_Training")
    logger.info("=============YOLO 火灾烟雾报警训练脚本启动=============")

    try:
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml(config_type="train")

        # 合并参数
        yolo_args, project_args = merge_args(args, yaml_config, mode="train")

        # 记录设备信息
        device_info = get_device_info()
        log_dict(logger = logger, title = "Device Info", data = device_info)

        # 记录参数
        arg_info = get_args_info(args)
        log_dict(logger = logger, title = "Args Info", data = arg_info)

        # 记录数据集信息（动态选择数据集分割）
        dataset_info = get_dataset_info(args.data)
        log_dict(logger = logger, title = "Dataset Info", data = dataset_info)

        # 初始化YOLO模型
        logger.info(f"初始化YOLO模型... 加载模型: {project_args.weights}")
        model_path = PRETRAINED_DIR / project_args.weights
        if not model_path.exists():
            logger.error(
                f"预训练模型不存在: {model_path}, " + 
                f"请将project_args.weights放入到{PRETRAINED_DIR}中"
            )
            raise ValueError("预训练模型不存在")
        model = YOLO(model_path)

        logger.info("开始训练...")
        run_results = run_training(model, yolo_args)

        trainer = model.trainer

        if run_results and hasattr(trainer, "save_dir"):
            
            setattr(run_results, "save_dir", str(trainer.save_dir))

            logger.info(f"训练结果保存在: {run_results.save_dir}")
            
            result_info = get_result_info(run_results)
            log_dict(logger, 'Result Info', result_info)

            copy_checkpoint_models(
                Path(run_results.save_dir), 
                project_args.weights, 
                logger
            )

            rename_logfile(logger, 'train')

        logger.info("=============YOLO 火灾烟雾检测训练脚本结束=============")

    except Exception as e:

        logger.error(f"训练错误: {e}", exc_info=True)
        return


if __name__ == "__main__":
    args = parse_args()

    log_level = args.log_level.upper()

    main_logger_instance = setup_logger(
        log_dir=LOGGING_DIR,
        log_type="train",
        log_level=log_level,
        logger_name="YOLO_Training",
        temp_log = True
    )

    main()