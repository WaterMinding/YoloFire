import sys
import yaml
import shutil
import logging
import argparse
from pathlib import Path
from copy import deepcopy
from logging import Logger
from argparse import Namespace

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# 将 yolo/ 目录添加到 sys.path 中
yolo_path = Path(__file__).parent.parents[0]
if not str(yolo_path) in sys.path:
    sys.path.insert(0, str(yolo_path))

from utils import setup_logger, log_dict
from utils import CONFIGS_DIR, LOGGING_DIR
from utils import CHECKPOINTS_DIR, ONNX_DIR


def yolo2onnx(args: Namespace, logger: Logger) -> Path:
    
    """
    将 YOLO 模型导出为 ONNX 格式

    Args:
        model_name (Path|str): 模型名称或路径
        args (Namespace): 命令行参数
        logger (Logger): 日志记录器
    
    Returns:
        Path: 导出的 ONNX 模型路径
    """

    # 获取待转换模型路径
    if args.model:

        model_name = args.model

    else:
        
        logger.error(f'缺少参数 --model')
        return None

    if isinstance(model_name, Path) and model_name.is_absolute():

        model_path = model_name
    
    else:

        model_path = CHECKPOINTS_DIR / model_name
    
    # 构造模型
    model = YOLO(model_path)

    # 导出为 ONNX 格式
    logger.info(f'正在将 {model_name} 导出为 ONNX 格式...')
    
    onnx_path = ONNX_DIR / f'{model_path.stem}.onnx'
    
    try:

        export_path = model.export(
            format = 'onnx',
            nms = not args.no_nms,
            half = args.half,
            int8 = args.int8,
            batch = args.batch,
            imgsz = args.imgsz,
            opset = args.opset,
            device = args.device,
            simplify = args.simplify,
        )

        shutil.move(export_path, onnx_path)
    
    except Exception as e:

        logger.error(f"导出ONNX模型失败: {e}")

        return None
    
    logger.info(f'导出ONNX模型成功: {onnx_path}')

    # 验证导出的模型
    logger.info('正在验证导出的模型...')

    try:

        model = ort.InferenceSession(onnx_path)

        input_name = model.get_inputs()[0].name

        fake_img = np.random.uniform(
            low = 0, high = 1,
            size = (args.batch, 3, args.imgsz, args.imgsz),
        ).astype(np.float32)

        outputs = model.run(None, {input_name: fake_img})

    except Exception as e:

        logger.error(f'导出的ONNX模型验证失败: {e}')
        logger.error(f'请检查ONNX模型文件: {onnx_path}')

        return None
    
    logger.info('导出的模型验证成功!')

    return onnx_path


def parse_args() -> Namespace:

    parser = argparse.ArgumentParser()
    add_argument = parser.add_argument

    add_argument('--model', type = str, help = '待转换模型名称')
    add_argument('--no_nms', action = 'store_false', help = '禁用NMS')
    add_argument('--half', action = 'store_true', help = '使用半精度浮点数')
    add_argument('--int8', action = 'store_true', help = '使用INT8量化')
    add_argument('--simplify', action = 'store_true', help = '简化ONNX模型')
    add_argument('--batch', type = int, default = 1, help = '批处理大小')
    add_argument('--imgsz', type = int, default = 640, help = '输入尺寸')
    add_argument('--opset', type = int, default = 11, help = 'opset版本')
    add_argument('--device', type = str, default = 'cpu', help = '设备类型')

    add_argument('--yaml', type = str, default = 'onnx.yaml', help='配置文件名')

    return parser.parse_args()


def merge_args(args: Namespace, logger: Logger) -> Namespace:

    """
    合并命令行参数和配置文件参数

    Args:
        args (Namespace): 命令行参数
        logger (Logger): 日志记录器

    Returns:
        Namespace: 合并后的参数
    """

    merged_args = deepcopy(args)

    # 合并配置文件参数（低优先级）
    if args.yaml is not None:

        logger.info(f'合并配置文件参数：{args.yaml}')

        yaml_path = Path(args.yaml)

        if not yaml_path.is_absolute():

            yaml_path = CONFIGS_DIR / yaml_path

        with open(yaml_path, 'r', encoding = 'utf-8') as f:

            yaml_args = yaml.safe_load(f)

        for key, value in yaml_args.items():

            # 防止配置文件混入错误参数
            if hasattr(args, key):

                setattr(merged_args, key, value)
    
    else:

        logger.warning('未指定配置文件，将使用默认配置')

    # 合并显式指定的命令行参数（高优先级）
    logger.info('合并显式指定的命令行参数')

    for key, value in vars(args).items():

        if f'--{key}' in sys.argv:

            setattr(merged_args, key, value)
    
    logger.info(f'参数合并完成！')
    
    log_dict(
        logger = logger,
        title = '合并后的参数',
        data = vars(merged_args)
    )
    
    return merged_args


if __name__ == '__main__':

    logger = setup_logger(
        log_dir = LOGGING_DIR,
        log_level = logging.INFO,
        log_type = 'yolo2onnx',
        logger_name = 'yolo2onnx',
    )

    args = parse_args()

    args = merge_args(args = args, logger = logger)

    onnx_path = yolo2onnx(args = args, logger = logger)
