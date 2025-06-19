import sys
import logging
from utils.paths import (
    YOLO_DIR,
    CONFIGS_DIR,
    RAW_IMAGES_DIR,
    RAW_LABELS_DIR,
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_IMAGES_DIR,
    VAL_LABELS_DIR,
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR,
    RUNS_DIR,
    SCRIPTS_DIR,
    WEIGHTS_DIR,
    ONNX_DIR,
    ENGINE_DIR,
    CHECKPOINTS_DIR,
    PRETRAINED_DIR,
    LOGGING_DIR
)
from utils.log_utils import setup_logger

# 确保项目根目录在系统路径中
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

# 配置初始化日志
logger = setup_logger(
    log_dir=LOGGING_DIR,
    log_type="initial",
    log_level=logging.INFO,
    logger_name="YOLO_Initialization"
)


def initialize_yolo_directories() -> None:
    """初始化YOLO项目目录结构并检查数据状态"""
    logger.info("开始初始化YOLO项目目录...")
    created_dirs = []
    existing_dirs = []
    data_status = []

    # 需要创建的目录列表
    directories = [
        CONFIGS_DIR,
        SCRIPTS_DIR,
        RUNS_DIR,
        WEIGHTS_DIR,
        ONNX_DIR,
        ENGINE_DIR,
        CHECKPOINTS_DIR,
        PRETRAINED_DIR,
        RAW_IMAGES_DIR,
        RAW_LABELS_DIR,
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR,
        TEST_IMAGES_DIR,
        TEST_LABELS_DIR
    ]

    # 创建目录
    for directory in directories:
        try:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"已创建目录: "
                            f"{directory.relative_to(YOLO_DIR)}")
                created_dirs.append(directory)
            else:
                logger.info(f"目录已存在: "
                            f"{directory.relative_to(YOLO_DIR)}")
                existing_dirs.append(directory)
        except Exception as e:
            logger.error(f"创建目录失败: "
                         f"{directory.relative_to(YOLO_DIR)} - {str(e)}")

    # 检查数据目录状态
    data_dirs = [
        (TRAIN_IMAGES_DIR, "训练图像"),
        (TRAIN_LABELS_DIR, "训练标签"),
        (VAL_IMAGES_DIR, "验证图像"),
        (VAL_LABELS_DIR, "验证标签"),
        (TEST_IMAGES_DIR, "测试图像"),
        (TEST_LABELS_DIR, "测试标签")
    ]

    for directory, desc in data_dirs:
        if not any(directory.iterdir()):
            status = (f"{directory.relative_to(YOLO_DIR)}: "
                      f"空目录，请放置{desc}数据")
            logger.warning(status)
            data_status.append(status)
        else:
            status = (f"{directory.relative_to(YOLO_DIR)}: "
                      f"已包含{desc}文件")
            logger.info(status)
            data_status.append(status)

    # 生成汇总报告
    logger.info("=" * 60)
    logger.info("YOLO项目初始化完成")
    logger.info(f"创建目录数量: {len(created_dirs)}")
    logger.info(f"现有目录数量: {len(existing_dirs)}")

    if data_status:
        logger.info("-" * 60)
        logger.info("数据目录状态:")
        for status in data_status:
            logger.info(f"  • {status}")

    logger.info("=" * 60)
    logger.info("请确保所有必要数据已正确放置，然后继续后续操作")


if __name__ == "__main__":
    initialize_yolo_directories()
