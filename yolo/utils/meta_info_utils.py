import yaml
import psutil
import logging
import platform
from pathlib import Path
from datetime import datetime
from argparse import Namespace

import torch
import pynvml
import ultralytics
from ultralytics.utils.metrics import DetMetrics

from .paths import CONFIGS_DIR

pynvml.nvmlInit()


def format_size(size: int) -> str:

    if size < 1024:

        return f"{size} B"

    elif size < 1024 ** 2:

        return f"{size / 1024:.2f} KB"

    elif size < 1024 ** 3:

        return f"{size / 1024 ** 2:.2f} MB"

    else:

        return f"{size / 1024 ** 3:.2f} GB"
    

def format_percentage(
    percentage: float,
    scale: bool = False,
) -> str:
    
    if scale:

        percentage = percentage * 100

    return f"{percentage:.2f}%"


def get_device_info() -> dict:

    logger = logging.getLogger("YOLO_Training")

    # 系统及环境信息
    general_dict = {}

    general_dict["OS_Type"] = platform.system()

    general_dict["OS_Version"] = platform.release()

    general_dict["Architecture"] = platform.machine()

    general_dict["Python_Version"] = platform.python_version()

    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    general_dict["System_Time"] = local_time
    
    # CPU信息
    cpu_dict = {}

    cpu_dict["Model"] = platform.processor()

    cpu_dict["Cores"] = psutil.cpu_count(logical = False)

    used_percentage = psutil.cpu_percent(interval = 1)

    cpu_dict["Usage_Percentage"] = format_percentage(used_percentage)

    # 内存信息
    memory_dict = {}

    ram_size = psutil.virtual_memory().total

    available_szie = psutil.virtual_memory().available

    used_percentage = psutil.virtual_memory().percent

    used_percentage = format_percentage(used_percentage)

    memory_dict["Total_RAM"] = format_size(ram_size)

    memory_dict["Available_RAM"] = format_size(available_szie)

    memory_dict["Used_RAM_Percentage"] = used_percentage

    # GPU信息
    gpu_dict = {}

    gpu_dict["CUDA_Available"] = torch.cuda.is_available()

    if gpu_dict["CUDA_Available"]:
    
        gpu_dict["Pytorch_CUDA_Version"] = torch.version.cuda

        gpu_dict["Nvidia_Driver_Version"] = pynvml.nvmlSystemGetDriverVersion()

        gpu_dict["Device_Count"] = pynvml.nvmlDeviceGetCount()

        gpu_dict["Selected_Devices"] = torch.cuda.device_count()

        gpu_dict["Details"] = []

        for i in range(gpu_dict["Device_Count"]):

            detail_dict = {}

            detail_dict["Index"] = i

            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            detail_dict["Model"] = pynvml.nvmlDeviceGetName(handle)

            detail_dict["Total_VRAM"] = format_size(
                pynvml.nvmlDeviceGetMemoryInfo(handle).total
            )

            detail_dict["CUDA_Capability"] = \
                pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            
            used_percentage = \
                pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            
            used_percentage = format_percentage(used_percentage)
            
            detail_dict["Usage_Percentage"] = used_percentage

            gpu_dict["Details"].append(detail_dict)
    
    else:

        logger.warning("Nvidia Driver not found")
    
    env_dict = {}

    env_dict["Pytorch_Version"] = torch.__version__

    try:
        import onnxruntime
        env_dict["ONNX_Runtime_Version"] = \
            onnxruntime.__version__
    except:
        env_dict["ONNX_Runtime_Version"] = "Not Installed"
        logger.warning("ONNX Runtime is not installed")
    
    try:
        import tensorrt
        env_dict["TensorRT_Version"] = \
            tensorrt.__version__
    except:
        env_dict["TensorRT_Version"] = "Not Installed"
        logger.warning("TensorRT_Version is not installed")

    env_dict["Cuda_Env"] = torch.version.cuda

    env_dict["Cudnn_Version"] = torch.backends.cudnn.version()

    env_dict["Ultralytics_Version"] = ultralytics.__version__

    return {
        "General": general_dict,
        "CPU": cpu_dict,
        "Memory": memory_dict,
        "GPU": gpu_dict,
        "Environment": env_dict,
    }


def get_args_info(args: Namespace) -> dict:

    return vars(args)


def get_dataset_info(yaml_name: str) -> dict:

    yaml_path = CONFIGS_DIR / yaml_name

    with open(yaml_path, "r", encoding="utf-8") as f:

        dataset_info = yaml.safe_load(f)

    data_info = {}
    
    data_info['Class_Count'] = dataset_info['nc']
    data_info['Class_Names'] = dataset_info['names']
    
    train_label = Path(dataset_info['train']).parent / 'labels'
    val_label = Path(dataset_info['val']).parent / 'labels'
    test_label = Path(dataset_info['test']).parent / 'labels'

    train_count = len(list(train_label.glob('*.txt')))
    val_count = len(list(val_label.glob('*.txt')))
    test_count = len(list(test_label.glob('*.txt')))

    data_info['Train_Count'] = train_count
    data_info['Val_Count'] = val_count
    data_info['Test_Count'] = test_count
    data_info['Config_File'] = yaml_path
    data_info['Train_Path'] = dataset_info['train']
    data_info['Val_Path'] = dataset_info['val']
    data_info['Test_Path'] = dataset_info['test']

    return data_info


def get_result_info(results: DetMetrics):

    metrics = results.results_dict

    speed = results.speed

    fitness = float(results.fitness)

    save_dir = results.save_dir \
        if hasattr(results, 'save_dir') else None

    task = results.task

    total_time = sum(speed.values())

    # 准备返回的字典
    results_dict = {
        "task": task,
        "speed": {
            "preprocess(ms)": speed["preprocess"],
            "inference(ms)": speed["inference"],
            "loss(ms)": speed["loss"],
            "postprocess(ms)": speed["postprocess"],
            "total_processing_per_image(ms)": total_time 
        },
        "metrics":{
            "fitness": fitness,
            "precision": float(metrics["metrics/precision(B)"]),
            "recall": float(metrics["metrics/recall(B)"]),
            "mAP@0.5": float(metrics["metrics/mAP50(B)"]),
            "mAP@0.5:0.95": float(metrics["metrics/mAP50-95(B)"])
        },
        "class_mAP50-95":{},
        "save_dir": save_dir or '验证模式不提供保存路径',
        "timestamp": datetime.now().isoformat()
    }

    return results_dict


