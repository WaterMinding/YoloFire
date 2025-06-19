#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :config_utils.py
# @Time      :2025/6/12 11:20:19
# @Author    :雨霓同学
# @Project   :SafeH
# @Function  :提供配置文件操作工具
from pathlib import  Path
import sys
import argparse

current_script_path = Path(__file__).parent.parent # yolo根目录
utils_path = current_script_path / "utils"  # utils目录
if str(current_script_path) not in sys.path:
    sys.path.insert(0, str(current_script_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

import yaml
from utils.paths import CONFIGS_DIR, RUNS_DIR
from utils.configs import COMMENTED_TRAIN_CONFIG, DEFAULT_TRAIN_CONFIG
from utils.configs import COMMENTED_VAL_CONFIG, DEFAULT_VAL_CONFIG
from utils.configs import COMMENTED_INFER_CONFIG, DEFAULT_INFER_CONFIG

VALID_YOLO_TRAIN_ARGS = set(DEFAULT_TRAIN_CONFIG.keys()) # 只包含官方参数
VALID_YOLO_VAL_ARGS = set(DEFAULT_VAL_CONFIG.keys())
VALID_YOLO_INFER_ARGS = set(DEFAULT_INFER_CONFIG.keys())

BOOLEAN_PARAMS = {
    key for config in (DEFAULT_INFER_CONFIG,  DEFAULT_VAL_CONFIG, DEFAULT_INFER_CONFIG)
    for key, value in config.items() if isinstance(value, bool)
}

import logging

# 配置 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志级别为 INFOlogger.setLevel(logging.INFO)  # 设置日志级别为 INFO

def load_yaml(config_type='train'):
    """
    加载YAML配置文件
    :param config_type: 配置文件类型，默认为train',可以设置为VaL, Infer
    :return: 配置文件内容,解析后的配置字典
    """
    config_path = CONFIGS_DIR / f"{config_type}.yaml"
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path},尝试生产默认的配置文件")
        if config_type in ["train", "val", "infer"]:
            try:
                logger.info(f"生成默认的配置文件: {config_path}")
                config_path.parent.mkdir(parents=True, exist_ok=True)
                generate_default_config(config_type=config_type)
            except Exception as e:
                logger.error(f"生成默认配置文件失败: {e}")
                raise FileNotFoundError(f"无法生成默认的配置文件: {e}")
        else:
            logger.error(f"未知的配置类型: {config_type}")
            raise ValueError(f"不支持的配置类型: {config_type}, 仅支持 【train, val, infer】三种配置文件生成")
    
    try:
        print(f"开始加载配置文件: {config_path}")
        logger.info(f"开始加载配置文件: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"已加载配置文件: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"解析配置文件失败: {e}")
        raise
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

def generate_default_config(config_type):
    """
    生成默认的配置文件
    :param config_type: 配置文件的类型
    :return:
    """
    config_path = CONFIGS_DIR / f"{config_type}.yaml"
    if config_type == "train":
        config = COMMENTED_TRAIN_CONFIG
    elif  config_type == "val":
        config = COMMENTED_VAL_CONFIG
    elif config_type == "infer":
        config = COMMENTED_INFER_CONFIG
    else:
        logger.error(f"未知的配置类型: {config_type}")
        raise ValueError(f"不支持的配置类型: {config_type}, 仅支持 【train, val, infer】三种配置文件生成")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config)
    logger.info(f"生成默认 {config_type} 配置文件成功: {config_path}")

# 假设已经存在命令行参数args,
def merge_args(args, yaml_config, mode='train'):
    """
    合并命令行参数、YAML 配置文件参数和默认参数，按优先级 CLI > YAML > 默认值。

    Args:
        args: 通过 argparse 解析的命令行参数（argparse.Namespace）。
        yaml_config: 从 YAML 文件加载的配置参数（dict）。
        mode: 运行模式，支持 'train'（训练）、'val'（验证）、'infer'（推理）。

    Returns:
        tuple: (yolo_args, project_args)
            - yolo_args: 包含 YOLO 官方参数的命名空间（argparse.Namespace）。
            - project_args: 包含所有项目参数的命名空间，包括来源标记（argparse.Namespace）。

    Raises:
        ValueError: 如果模式无效或参数验证失败。
    """
    # logger = logging.getLogger("YOLO_Training")

    # 1. 确定运行模式和相关配置
    if mode == 'train':
        valid_args = VALID_YOLO_TRAIN_ARGS
        default_config = DEFAULT_TRAIN_CONFIG
    elif mode == 'val':
        valid_args = VALID_YOLO_VAL_ARGS
        default_config = DEFAULT_VAL_CONFIG
    elif mode == 'infer':
        valid_args = VALID_YOLO_INFER_ARGS
        default_config = DEFAULT_INFER_CONFIG
    else:
        logger.error(f"无效模式: {mode}，支持 'train', 'val', 'infer'")
        raise ValueError(f"无效模式: {mode}")

    # 2. 初始化参数存储
    project_args = argparse.Namespace()  # 存储所有项目参数
    yolo_args = argparse.Namespace()    # 存储 YOLO 官方参数
    merged_params = default_config.copy()  # 合并参数的字典，以默认配置为最低优先级

    # 3. 合并 YAML 参数（优先级高于默认值）
    if hasattr(args, 'use_yaml') and args.use_yaml and yaml_config:
        for key, value in yaml_config.items():
            # 处理布尔参数（将字符串 'true'/'false' 转为 bool）
            if key in BOOLEAN_PARAMS and isinstance(value, str):
                value = value.lower() == 'true'
            # 处理 None（将字符串 'none' 转为 Python 的 None）
            elif isinstance(value, str) and value.lower() == 'none':
                value = None
            # 处理 classes（将字符串 '0,1,2' 转为列表 [0, 1, 2]）
            elif key == 'classes' and isinstance(value, str) and value:
                try:
                    value = [int(i.strip()) for i in value.split(',')]
                except ValueError:
                    logger.warning(f"YAML 中 'classes' 参数 '{value}' 格式不正确，保留原值")
            merged_params[key] = value
        logger.debug(f"合并 YAML 参数后: {merged_params}")

    # 4. 合并命令行参数（最高优先级）
    # 处理预定义的命令行参数（通过 parser.add_argument 定义）
    cmd_args = {k: v for k, v in vars(args).items() if k != 'extra_args' and v is not None}
    for key, value in cmd_args.items():
        # 处理 classes
        if key == 'classes' and isinstance(value, str):
            if value.lower() == 'none':
                value = None
            elif value:
                try:
                    value = [int(i.strip()) for i in value.split(',')]
                except ValueError:
                    logger.warning(f"命令行 'classes' 参数 '{value}' 格式不正确，保留原值")
        # 处理布尔参数
        elif key in BOOLEAN_PARAMS and isinstance(value, str):
            value = value.lower() == 'true'
        merged_params[key] = value
        setattr(project_args, f"{key}_specified", True)  # 标记 CLI 参数来源

    # 处理动态参数（extra_args，未在 parser.add_argument 中定义）
    if hasattr(args, 'extra_args'):
        if len(args.extra_args) % 2 != 0:
            logger.error("额外参数格式错误，必须成对出现（如 --key value）")
            raise ValueError("额外参数格式错误")
        for i in range(0, len(args.extra_args), 2):
            key = args.extra_args[i].lstrip('--')
            value = args.extra_args[i + 1]
            try:
                # 尝试类型转换
                if value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.lower() == 'none':
                    value = None
                elif key == 'classes' and value:
                    try:
                        value = [int(i.strip()) for i in value.split(',')]
                    except ValueError:
                        logger.warning(f"额外参数 'classes' '{value}' 格式不正确，保留原值")
            except ValueError:
                logger.warning(f"无法转换额外参数 {key} 的值 {value}")
            merged_params[key] = value
            setattr(project_args, f"{key}_specified", True)  # 标记 extra_args 来源

    # 5. 路径标准化（基于 utils.paths 模块）
    # 标准化 data 参数（通常为 YAML 配置文件路径）
    if 'data' in merged_params and merged_params['data']:
        data_path = Path(merged_params['data'])
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / data_path  # 假设 data.yaml 在 CONFIGS_DIR 下
        merged_params['data'] = str(data_path)
        # 验证路径存在性
        if not data_path.exists():
            logger.warning(f"数据集配置文件 '{data_path}' 不存在，请检查")
        logger.info(f"标准化数据集路径: {merged_params['data']}")

    # 标准化 project 参数（训练或推理结果保存目录）
    if 'project' in merged_params and merged_params['project']:
        project_path = Path(merged_params['project'])
        if not project_path.is_absolute():
            project_path = RUNS_DIR / project_path  # 假设 project 在 RUNS_DIR 下
        merged_params['project'] = str(project_path)
        # 验证目录可写性
        try:
            project_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"无法创建或写入 project 目录 '{project_path}'")
            raise ValueError(f"project 目录 '{project_path}' 不可写")
        logger.info(f"标准化 project 路径: {merged_params['project']}")

    # 6. 分离 yolo_args 和 project_args
    for key, value in merged_params.items():
        setattr(project_args, key, value)
        if key in valid_args:
            setattr(yolo_args, key, value)
        # 标记 YAML 参数来源（未被 CLI 或 extra_args 覆盖）
        if key in yaml_config and not hasattr(project_args, f"{key}_specified"):
            setattr(project_args, f"{key}_specified", False)

    # 7. 参数验证（确保参数符合 YOLO 要求）
    if mode == 'train':
        # 验证 epochs
        if hasattr(yolo_args, 'epochs') and (not isinstance(yolo_args.epochs, int) or yolo_args.epochs <= 0):
            logger.error("训练轮数 (epochs) 必须为正整数")
            raise ValueError("训练轮数 (epochs) 必须为正整数")
        # 验证 imgsz
        if hasattr(yolo_args, 'imgsz') and (not isinstance(yolo_args.imgsz, int) or yolo_args.imgsz <= 0 or yolo_args.imgsz % 8 != 0):
            logger.error("图像尺寸 (imgsz) 必须为正整数且为 8 的倍数")
            raise ValueError("图像尺寸 (imgsz) 必须为正整数且为 8 的倍数")
        # 验证 batch
        if hasattr(yolo_args, 'batch') and yolo_args.batch is not None and (not isinstance(yolo_args.batch, int) or yolo_args.batch <= 0):
            logger.error("批次大小 (batch) 必须为正整数或 None（自动批次）")
            raise ValueError("批次大小 (batch) 必须为正整数或 None")
        # 验证 data 配置文件
        if hasattr(yolo_args, 'data') and yolo_args.data and not Path(yolo_args.data).is_file():
            logger.error(f"数据集配置文件 '{yolo_args.data}' 不存在或不是文件")
            raise ValueError("数据集配置文件无效")
    elif mode == 'val':
        # 确保 split 参数存在
        # if not hasattr(yolo_args, 'split'):
        #     logger.warning("验证模式缺少 split 参数，设置为默认 'val'")
        #     setattr(yolo_args, 'split', 'val')
        #     setattr(project_args, 'split', 'val')
        # 验证 data 配置文件
        if hasattr(yolo_args, 'data') and yolo_args.data and not Path(yolo_args.data).is_file():
            logger.error(f"数据集配置文件 '{yolo_args.data}' 不存在或不是文件")
            raise ValueError("数据集配置文件无效")
    elif mode == 'infer':
        # 验证 model 文件
        if hasattr(yolo_args, 'model') and yolo_args.model and not Path(yolo_args.model).is_file():
            logger.error(f"模型文件 '{yolo_args.model}' 不存在或不是文件")
            raise ValueError("模型文件无效")

    logger.debug(f"yolo_args: {vars(yolo_args)}")
    logger.debug(f"project_args: {vars(project_args)}")
    return yolo_args, project_args

def rename_logfile(loggers,save_dir, model_name,encoding='utf-8'):
    """
    重命名日志文件，使用训练目录的名称，train1, train2, val1, val2
    :param loggers: 日志记录器对象
    :param save_dir: 输出的目录
    :param model_name: 模型的名称
    :param encoding: 编码
    :return:
    """
    for handler in loggers.handlers:
        # 检查当前处理器
        if isinstance(handler, logging.FileHandler):
            # 获取当前处理器的文件名
            old_log_file = Path(handler.baseFilename)
            timestamp = old_log_file.stem.split('_')[1]
            train_prefix = Path(save_dir).name
            new_log_file = old_log_file.parent / f"{train_prefix}_{timestamp}_{model_name}.log"
            handler.close()
            loggers.removeHandler(handler)

            if old_log_file.exists():
                old_log_file.rename(new_log_file)
                loggers.info(f"日志文件已重命名: {new_log_file}")
            new_handler = logging.FileHandler(new_log_file, encoding=encoding)

            new_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            loggers.addHandler(new_handler)
            break


if __name__ == '__main__':
    load_yaml(config_type='train')
