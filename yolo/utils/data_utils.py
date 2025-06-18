import yaml
from pathlib import Path
import logging
from .log_utils import log_dict


def gen_data_config(data_dir: Path, config_dir: Path,
                    logger: logging.Logger) -> Path:
    """
    生成YOLO数据配置文件(data.yaml)
    :param data_dir:Path 数据集根目录
    :param config_dir:Path 配置文件存放目录
    :param logger:logging.Logger 日志记录器实例
    :return:Path: 生成的配置文件路径
    """
    # 确保配置目录存在
    config_dir.mkdir(parents=True, exist_ok=True)

    # 验证数据集目录结构
    required_dirs = {
        "train": data_dir / "train" / "images",
        "val": data_dir / "val" / "images",
        "test": data_dir / "test" / "images"
    }

    # 检查必要目录是否存在
    for name, path in required_dirs.items():
        if not path.exists():
            logger.error(f"数据集目录结构错误: 缺少 {name} 目录 ({path})")
            raise FileNotFoundError(f"Missing required directory: {path}")

    # 检查测试集是否存在（可选）
    # test_dir = data_dir / "test" / "images"
    # test_path = str(test_dir) if test_dir.exists() else None

    # 获取类别信息
    classes_path = data_dir / "classes.txt"
    if not classes_path.exists():
        logger.warning(f"未找到类别文件: {classes_path}，将使用空类别列表")
        class_names = []
    else:
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [
                line.strip() for line in f.readlines() if line.strip()]

    # 构建数据配置字典
    data_config = {
        "path": str(data_dir),  # 数据集根目录
        "train": str(required_dirs["train"]),  # 训练集路径
        "val": str(required_dirs["val"]),  # 验证集路径
        "test": str(required_dirs["test"]),  # 测试集路径
        # "test": test_path,  # 测试集路径（可选）
        "nc": len(class_names),  # 类别数量
        "names": class_names,  # 类别名称列表
        "notes": "自动生成的YOLO数据集配置文件"
    }

    # 保存配置文件
    config_path = config_dir / "data.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=None, allow_unicode=True, sort_keys=False)

    logger.info(f"已生成数据集配置文件: {config_path}")

    # 记录配置文件内容
    log_dict(logger, "数据集配置文件内容", data_config)

    return config_path


def check_data(data_dir: Path, logger: logging.Logger) -> bool:
    """
    验证数据集目录结构并确保数据完整性
    :param data_dir:Path 数据集根目录
    :param logger:logging.Logger 日志记录器实例
    :return:bool: 数据集是否通过验证
    """
    # 定义支持的图片和标签文件扩展名
    IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    LABEL_EXTS = ['.txt']

    # 检查数据集子目录结构
    valid_splits = ['train', 'val', 'test']
    all_classes = set()
    validation_passed = True

    # 记录验证开始
    logger.info(f"开始验证数据集目录结构: {data_dir}")
    log_dict(logger, "数据集验证标准", {
        "目录要求": "必须包含 train, val, test 子目录",
        "文件要求": "每个子目录下必须有 images 和 labels 子目录",
        "文件对齐": "图像和标签文件必须一一对应",
        "类别文件": "根目录下应有 classes.txt 文件"
    })

    # 1. 检查必需子目录是否存在
    for split in valid_splits:
        split_path = data_dir / split
        if not split_path.exists():
            logger.warning(f"缺失 {split} 目录: {split_path}")
            validation_passed = False
            continue

        # 2. 检查 images 和 labels 子目录
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        if not images_dir.exists():
            logger.error(f"{split} 目录缺少 images 子目录: {images_dir}")
            validation_passed = False
        if not labels_dir.exists():
            logger.error(f"{split} 目录缺少 labels 子目录: {labels_dir}")
            validation_passed = False

        # 3. 检查目录不为空
        if images_dir.exists():
            image_files = [f for ext in IMAGE_EXTS for
                           f in images_dir.glob(f"*{ext}")]
            if not image_files:
                logger.error(f"{split}/images 目录为空")
                validation_passed = False

        if labels_dir.exists():
            label_files = [f for ext in LABEL_EXTS for
                           f in labels_dir.glob(f"*{ext}")]
            if not label_files:
                logger.error(f"{split}/labels 目录为空")
                validation_passed = False

        # 4. 检查文件一一对应
        if images_dir.exists() and labels_dir.exists():
            image_stems = {f.stem for f in image_files}
            label_stems = {f.stem for f in label_files}

            # 检查缺失的标签文件
            missing_labels = image_stems - label_stems
            if missing_labels:
                logger.error(f"{split} 目录中 "
                             f"{len(missing_labels)} 个图像缺少对应的标签文件")
                for i, stem in enumerate(list(missing_labels)[:5]):
                    logger.error(f"  缺失标签: {stem} (显示前5个)")
                validation_passed = False

            # 检查多余的标签文件
            extra_labels = label_stems - image_stems
            if extra_labels:
                logger.warning(f"{split} 目录中有 "
                               f"{len(extra_labels)} 个标签文件没有对应的图像")
                for i, stem in enumerate(list(extra_labels)[:5]):
                    logger.warning(f"  多余标签: {stem} (显示前5个)")

            # 5. 收集所有类别信息
            for label_file in label_files:
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                all_classes.add(class_id)
                except Exception as e:
                    logger.error(f"读取标签文件 {label_file.name} 失败: {e}")
                    validation_passed = False

    # 6. 检查类别文件
    classes_file = data_dir / "classes.txt"
    if not classes_file.exists():
        logger.warning(f"未找到类别文件: {classes_file}")

        if all_classes:
            # 自动生成类别文件
            sorted_classes = sorted(all_classes)
            max_class = max(sorted_classes) if sorted_classes else 0

            # 确保类别ID连续
            expected_classes = set(range(max_class + 1))
            missing_classes = expected_classes - all_classes
            if missing_classes:
                logger.error(f"类别ID不连续，缺少以下ID: {sorted(missing_classes)}")
                validation_passed = False

            # 生成类别名称（使用占位符）
            class_names = [f"class_{i}" for i in sorted_classes]

            with open(classes_file, 'w', encoding='utf-8') as f:
                for name in class_names:
                    f.write(f"{name}\n")

            logger.info(f"已自动生成类别文件: {classes_file}")
            logger.info(f"检测到 {len(class_names)} 个类别，使用占位符名称")
        else:
            logger.error("无法自动生成类别文件，未检测到任何类别ID")
            validation_passed = False
    else:
        # 验证现有类别文件
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]

            num_classes = len(class_names)
            logger.info(f"找到类别文件: {classes_file}, 包含 {num_classes} 个类别")

            # 检查类别ID是否在有效范围内
            if all_classes:
                max_id = max(all_classes)
                if max_id >= num_classes:
                    logger.error(f"标签文件中存在无效类别ID: "
                                 f"{max_id} (最大应为 {num_classes - 1})")
                    validation_passed = False
        except Exception as e:
            logger.error(f"读取类别文件失败: {e}")
            validation_passed = False

    # 生成验证总结报告
    if validation_passed:
        logger.info("=" * 60)
        logger.info("数据集验证通过! 所有检查项符合要求")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("数据集验证失败! 请检查上述错误信息")
        logger.error("=" * 60)

    return validation_passed
