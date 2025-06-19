from pathlib import Path

YOLO_DIR = Path(__file__).parent.parent


CONFIGS_DIR = YOLO_DIR / "configs"

DATA_DIR = YOLO_DIR / "dataset"

RAW_DATA_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
RAW_LABELS_DIR = RAW_DATA_DIR / "labels"

TRAIN_DATA_DIR = DATA_DIR / "train"
TRAIN_IMAGES_DIR = TRAIN_DATA_DIR / "images"
TRAIN_LABELS_DIR = TRAIN_DATA_DIR / "labels"

VAL_DATA_DIR = DATA_DIR / "val"
VAL_IMAGES_DIR = VAL_DATA_DIR / "images"
VAL_LABELS_DIR = VAL_DATA_DIR / "labels"

TEST_DATA_DIR = DATA_DIR / "test"
TEST_IMAGES_DIR = TEST_DATA_DIR / "images"
TEST_LABELS_DIR = TEST_DATA_DIR / "labels"


LOGGING_DIR = YOLO_DIR / "logs"

RUNS_DIR = YOLO_DIR / "runs"

SCRIPTS_DIR = YOLO_DIR / "scripts"

WEIGHTS_DIR = YOLO_DIR / "weights"

ONNX_DIR = WEIGHTS_DIR / "onnx"
ENGINE_DIR = WEIGHTS_DIR / "engine"
CHECKPOINTS_DIR = WEIGHTS_DIR / "checkpoints"
PRETRAINED_DIR = WEIGHTS_DIR / "pretrained"


