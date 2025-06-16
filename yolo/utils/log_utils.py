import os
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(log_dir: Path, log_type: str,
                 log_level: int = logging.INFO,
                 logger_name: str = "YOLO_Logger",
                 temp_log: bool = False) -> logging.Logger:
    f"""
    配置日志，保存到logs/{log_type}目录
    :param log_dir: Path 存放日志文件的根目录
    :param log_type: str 日志类型（用于子目录和文件名）
    :param log_level: int 日志记录等级（默认为INFO）
    :param logger_name: str 日志记录器名称（默认为YOLO_Logger）
    :param temp_log: bool 是否为临时日志（默认为False）
    :return: logging.Logger: 配置好的日志记录器
    """

    log_type_dir = log_dir / log_type
    log_type_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "temp" if temp_log else log_type
    log_filename = f"{prefix}-{timestamp}.log"
    log_filepath = log_type_dir / log_filename

    # 使用传入的logger_name来获取和配置Logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # 避免重复添加处理器
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    logger.info(f"日志初始化完成 | 记录器: {logger_name}")
    logger.info(f"日志目录: {log_type_dir}")
    logger.info(f"日志文件: {log_filename} (UTF-8编码)")
    logger.info(f"日志类型: {'临时日志' if temp_log else log_type}")
    logger.info(f"日志级别: {logging.getLevelName(log_level)}")

    return logger


def log_dict(logger: logging.Logger, title: str, data: dict) -> None:
    """
    负责将一个字典以格式化的方式记录为日志
    :param logger: logging.Logger 用于记录的logger
    :param title: str 日志记录的标题
    :param data: dict 要记录的字典数据
    """
    # 标题分隔线
    title_line = f" {title} ".center(60, '=')
    logger.info(title_line)

    # 递归处理字典内容
    def _log_dict_content(data, indent=0):

        indent_str = ' ' * indent * 4  # 4空格缩进

        for key, value in data.items():
            # 处理嵌套字典
            if isinstance(value, dict):
                logger.info(f"{indent_str}{key}:")
                _log_dict_content(value, indent + 1)

            # 处理包含字典的列表
            elif (isinstance(value, list) and
                  any(isinstance(item, dict) for item in value)):
                logger.info(f"{indent_str}{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        logger.info(f"{indent_str}  - 条目 #{i + 1}:")
                        _log_dict_content(item, indent + 2)
                    else:
                        logger.info(f"{indent_str}  - {item}")

            # 处理普通键值对
            else:
                # 格式化列表为可读字符串
                if isinstance(value, list):
                    value_str = ', '.join(map(str, value))
                    logger.info(f"{indent_str}{key}: [{value_str}]")
                else:
                    logger.info(f"{indent_str}{key}: {value}")

    # 开始处理字典内容
    _log_dict_content(data)
    logger.info("=" * 60)


def rename_logfile(logger: logging.Logger, new_logtype: str) -> Path:
    """
    重命名日志文件并将临时日志转为正式日志
    :param logger: logging.Logger 日志记录器实例
    :param new_logtype: str 新的日志类型名称
    :return: Path: 新的日志文件路径
    """
    # 查找文件处理器
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break

    if not file_handler:
        logger.warning("未找到文件日志处理器，无法重命名日志文件")
        return None

    # 获取当前日志文件路径
    old_path = Path(file_handler.baseFilename)

    # 验证是否为临时日志（文件名以"temp-"开头）
    if not old_path.name.startswith("temp-"):
        logger.warning(f"日志文件 {old_path.name} 不是临时日志，无法重命名")
        return old_path

    # 生成新文件名（替换temp为new_logtype）
    new_filename = old_path.name.replace("temp-", f"{new_logtype}-", 1)
    new_path = old_path.with_name(new_filename)

    # 关闭当前文件处理器
    file_handler.close()
    logger.removeHandler(file_handler)

    # 重命名文件
    try:
        os.rename(old_path, new_path)
        logger.info(f"已重命名日志文件: {old_path} -> {new_path}")
    except OSError as e:
        logger.error(f"重命名日志文件失败: {e}")
        # 恢复原始处理器
        logger.addHandler(file_handler)
        return old_path

    # 创建新文件处理器
    new_handler = logging.FileHandler(new_path, encoding='utf-8')
    new_handler.setFormatter(file_handler.formatter)
    logger.addHandler(new_handler)

    # 记录重命名信息
    logger.info(f"日志类型已更新: temp -> {new_logtype}")
    logger.info(f"当前日志文件: {new_path}")

    return new_path
