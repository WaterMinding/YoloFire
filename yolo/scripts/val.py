import sys
import logging
import argparse
from pathlib import Path

# 将yoloserver根目录添加到sys.path中
current_path = Path(__file__).parent
yoloserver_path = current_path.parents[0]
if str(yoloserver_path) not in sys.path:
    sys.path.insert(0, str(yoloserver_path))

from ultralytics import YOLO

from utils import LOGGING_DIR
from utils import CHECKPOINTS_DIR
from utils import get_dataset_info
from utils import get_device_info
from utils import get_args_info
from utils import get_result_info
from utils import merge_args
from utils import rename_logfile
from utils import setup_logger
from utils import load_yaml
from utils import log_dict


def parse_args():

    parser = argparse.ArgumentParser(description="YOLO Validation")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--weights", type=str, default="train8-20250613_143050-yolov8n-last.pt", help="Path to weights.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device to use")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    parser.add_argument("--split", type=str, default="test",choices = ["val", "test"], help="Split to use")

    parser.add_argument("--log_encoding", type=str, default = "utf-8", help="Log encoding")
    parser.add_argument("--log_level", type=str, default = "INFO", help="Log level")
    parser.add_argument("--use_yaml", type=bool, default = True, help="Use yaml")

    return parser.parse_args()

def validate_model(model, yolo_args):

    results = model.val(**vars(yolo_args))

    return results

def main():

    args = parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    model_name = Path(args.weights).stem

    logger = setup_logger(
        log_dir = LOGGING_DIR,
        log_type = "val",
        log_level = log_level,
        logger_name = 'val',
        temp_log = True
    )


    logger.info("YOLO火灾烟雾检测验证开始".center(60, "="))

    try:
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml(config_type = "val")
    except Exception as e:

        logger.error(f"加载yaml配置文件失败: {e}")
    
    # 合并参数
    yolo_args, project_args = merge_args(args, yaml_config, "val")

    device_info = get_device_info()
    log_dict(logger, 'Device Info', device_info)

    param_info = get_args_info(args)
    log_dict(logger, 'Args Info', param_info)

    dataset_info = get_dataset_info(args.data)
    log_dict(logger, 'Dataset Info', dataset_info)

    model_path = Path(args.weights)

    if not model_path.is_absolute():

        model_path = CHECKPOINTS_DIR / project_args.weights
    
    if not model_path.exists():

        logger.error(f"权重文件不存在: {model_path}")

        return None
    
    model = YOLO(model_path, project_args)

    run_results = validate_model(model, yolo_args)

    result_info = get_result_info(run_results)
    log_dict(logger, 'Result Info', result_info )

    log_path = rename_logfile(
        logger = logger,
        new_logtype = "val"
    )

    logger.info("YOLO火灾烟雾检测验证完成".center(60, "="))
    
if __name__ == "__main__":

    main()
