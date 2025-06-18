import sys
import logging
from pathlib import Path

# 设置项目根目录
current_script_path = Path(__file__).resolve()
yoloserver_root_path = current_script_path.parents[1]
utils_root_path = yoloserver_root_path / "utils"

# 添加必要的路径到系统路径
if str(yoloserver_root_path) not in sys.path:
    sys.path.insert(0, str(yoloserver_root_path))
if str(utils_root_path) not in sys.path:
    sys.path.insert(1, str(utils_root_path))

# 导入自定义工具
from utils import (
    setup_logger,
    log_dict,
    rename_logfile,
    DATA_DIR,
    CONFIGS_DIR,
    LOGGING_DIR
)
from utils.data_utils import gen_data_config, check_data


def prepare_dataset(logger: logging.Logger) -> bool:
    """
    数据集准备主流程
    :param logger:logging.Logger 配置好的日志记录器
    :return:bool: 数据集准备是否成功
    """
    # 记录流程开始
    logger.info("=" * 80)
    logger.info("开始数据集准备流程".center(80))
    logger.info("=" * 80)

    # 第一阶段：数据集验证
    logger.info("")
    logger.info("阶段 1: 数据集完整性验证".center(80, "-"))
    logger.info("验证数据集目录结构、文件对齐和类别文件")

    validation_result = check_data(DATA_DIR, logger)

    if not validation_result:
        logger.error("数据集验证失败！请根据上述错误修复问题")
        return False

    logger.info("✓ 数据集验证通过")

    # 第二阶段：生成配置文件
    logger.info("")
    logger.info("阶段 2: 生成数据配置文件".center(80, "-"))
    logger.info(f"将在 {CONFIGS_DIR} 目录生成 data.yaml 配置文件")

    try:
        config_path = gen_data_config(DATA_DIR, CONFIGS_DIR, logger)
        logger.info(f"✓ 配置文件生成成功: {config_path}")

        # 记录配置文件内容
        with open(config_path, "r", encoding="utf-8") as f:
            config_content = f.read()
        logger.info("配置文件内容预览:")
        logger.info("-" * 80)
        logger.info(config_content)
        logger.info("-" * 80)

        return True
    except Exception as e:
        logger.exception(f"配置文件生成失败: {str(e)}")
        return False


def main():
    """脚本主函数"""
    # 初始化日志记录器
    logger = setup_logger(
        log_dir=LOGGING_DIR,
        log_type="prepare_data",
        logger_name="DatasetPreparation",
        temp_log=True  # 初始为临时日志
    )

    try:
        # 执行数据集准备流程
        success = prepare_dataset(logger)

        # 根据结果记录最终状态
        logger.info("")
        logger.info("最终结果".center(80, "="))
        if success:
            logger.info("✓ 数据集准备成功！可以开始训练")
            # 将临时日志转为正式日志
            new_log_path = rename_logfile(logger, "prepare_data")
            logger.info(f"完整日志已保存至: {new_log_path}")
        else:
            logger.error("✗ 数据集准备失败！请检查错误信息")
            # 保留临时日志用于调试
            logger.info("临时日志保留在: " + logger.handlers[0].baseFilename)

        logger.info("=" * 80)

        # 返回适当的退出码
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.exception("脚本执行过程中发生未处理的异常")
        sys.exit(2)


if __name__ == "__main__":
    main()
