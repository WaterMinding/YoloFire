# YoloFire烟雾火焰检测系统总体需求文档

## 文档信息

- **文档编号**：YOLOFIRE-001
- **版本**：1.0
- **作者**：WaterMinding

- **创建日期**：2025年6月15日
- **更新日期**：2025年6月15日
- **分发范围**：数据工程师、算法工程师、测试工程师、UI工程师

---

## 1. 项目背景

- 本项目旨在开发一个基于 Ultralytics YOLOv8 的烟雾火焰检测系统，用于实时检测局部区域是否存在火灾隐患，识别烟雾、火焰。系统涵盖数据集转换、模型训练、模型验证、模型推理及推理结果美化五个核心模块，目标是提供自动化、标准化、高效的工作流程，减少手动操作，提高检测准确性和结果展示的直观性，为消防安全管理提供技术支持。

---

## 2. 目标

### 2.1 业务目标

1. 提供对图片、视频、实时摄像头等多种数据源的检测功能。

2. 提供对火焰、烟雾目标的自动警示功能（声音、画面、远程通信）。

3. 提供检测数据的UI界面展示功能（目标在图像中的位置、类型、置信度、FPS等）。

4. 提供多个模型选项，根据硬件配置自适应默认模型

5. 提供检测日志的实时记录（包含时间戳、数据源、目标数据等等信息）。

### 2.2  技术目标

- 开发数据集转换模块，将 Pascal VOC XML 格式转换为 YOLO 格式，生成标准目录结构和 `data.yaml`。
- 实现模型训练和验证脚本，支持命令行和 YAML 参数配置，统一日志和结果保存。
- 提供灵活的推理脚本，支持流式处理和可视化输出，集成语音提醒功能。
- 确保跨平台兼容性（Windows/Linux）、健壮的错误处理和统一的路径管理。
- 集成统一的日志系统，记录所有关键操作、设备信息和耗时统计。

### 2.3 交付目标

- 交付脚本 `scripts/*` `initialize_project.py`。
- 交付 `utils` 工具包，包含路径、日志、数据处理、验证等模块。
- 提供 README 和测试报告，说明各模块用法和执行顺序。
- 完成Git提交，develop分支。

## 3. 功能需求

### 3.1 YOLO包初始化

- **FR-001：路径脚本（yolo/utils/paths.py）**

  - 项目完整目录结构如下：

  - 路径脚本只需记录`yolo/`之下的部分。

    ```bash
    YoloFire
    ├── yolo/
    │   ├── configs/
    │   ├── dataset/
    │   │   ├── train/
    │   │   │   ├── images/
    │   │   │   └── labels/
    │   │   ├── val/
    │   │   │   ├── images/
    │   │   │   └── labels/
    │   │   └── test/
    │   │       ├── images/
    │   │       └── labels/
    │   ├── logs/
    │   ├── runs/
    │   ├── scripts/
    │   ├── utils/
    │   └── weights/
    │       ├── onnx/
    │       ├── checkpoints/
    │       └── pretrained/
    ├── app/
    │   ├── ui/
    │   ├── logs/
    │   ├── onnx/
    │   ├── cores/
    │   └── resources/
    ├── test/
    │   ├── test_ui/
    │   └── test_yolo/
    └── docs/
    ```

  - 此处对部分路径名称做简要规定

    ```python
    YOLO_DIR = Path(__file__).parent.parent
    CONFIGS_DIR
    SCRIPTS_DIR
    RUNS_DIR
    DATA_DIR 
    WEIGHTS_DIR
    CHECKPOINTS_DIR
    PRETRAINED_DIR
    ONNX_DIR
    LOGGING_DIR
    TRAIN_DATA_DIR
    TRAIN_IMAGES_DIR
    TRAIN_LABELS_DIR
    TEST_DATA_DIR
    TEST_IMAGES_DIR
    TEST_LABELS_DIR
    VAL_DATA_DIR
    VAL_IMAGES_DIR
    VAL_LABELS_DIR
    ```

    

- **FR-002：日志记录工具（yolo/utils/log_utils.py）**

  - 提供 `setup_logger()` 函数，负责创建和初始化日志记录器
    - 日志以utf-8编码。
    - **日志名称格式为：{log_type{id}/"temp"}-{timestamp}**
    - 函数参数：
      - log_dir: Path 存放日志文件的根目录。
      - log_type: str 定义日志文件类型和存放的子目录。
      - log_level: int 定义日志最低记录等级。
      - logger_name: str 定义日志记录器的名称。
      - temp_log: bool 判定是否将日志文件作为暂存日志。
  - 提供 `log_dict()` 函数，负责将一个字典以格式化的方式记录为日志
    - 函数参数：
      - logger: logging.Logger 用于记录的logger。
      - title: str 这一次记录的标题。
    - 函数实现：
      - 首先用“=”包围着记录标题。
      - 其次处理字典内部键值对
        - 如果键值对的值不是字典或包含字典的列表，则按每行一个{key:value}格式输出。
        - 如果键值对的值是包含字典的列表，则先将其键输出，然后继续按照当前规则处理其值的元素。
        - 如果键值对的值是字典，则先将其键输出，然后继续按照当前规则处理值字典。
  - 提供 `rename_logfile()` 函数，负责更改给定logger的暂存日志文件名称，并更新其handler。
    - 函数参数：
      - logger: logging.Logger 需要更改日志文件名的logger。
      - new_logtype: str 要将temp替换为new_logtype。
    - 函数返回值：
      - 日志文件新路径：Path

- **FR-003：项目初始化脚本（yolo/initial_yolo.py）** 

  - 本脚本的日志类型为 **initial**
  - 同上，项目初始化脚本只需初始化`yolo/`之下的部分。
  - 如果初始化过程中发现数据不存在，则通过日志提醒用户安置数据。

### 3.2 数据处理

- **FR-004：数据集处理工具（yolo/utils/data_utils.py）**

  - 提供数据配置文件生成函数 `gen_data_config`

    - 函数参数
      - data_dir: Path 数据集目录
      - config_dir: Path 配置文件目录
      - logger: logging.Loger 日志记录器。

    - 函数的目的是提供data.yaml文件便于后续训练（data.yaml具体内容格式请自行查询）。 

    - 生成后日志记录data.yaml具体内容。

  - 提供数据集验证函数 `check_data`
    - 函数参数：
      - data_dir: Path 数据集目录
      - logger: logging.Loger 日志记录器。
    - 本函数的目的是确定`dataset/`目录下数据满足如下要求，不满足之处通过日志提醒：
      - 对于`dataset/`下的每一个子目录来说，都不能为空（图像和标签都存在）
      - 对于`dataset/`下的每一个子目录来说，images文件夹中应当只有图片（可能有多种文件类型），labels文件夹中应当只有txt文件。
      - 对于`dataset/`下的每一个子目录来说，图像文件和标签文件都应当对齐（达成一一对应）。
      - 检查`dataset/`目录下是否存在data.txt文件，如果没有，则统计数据中包含的类型，自动生成data.txt（该文件的具体形式请自行上网查询）。

- **FR-005：数据集准备脚本（yolo/scripts/prepare_data.py）**
  - 本脚本的日志类型为 **prepare_data**
  - 本脚本的目的是调用上述两个功能，完成对data.yaml的生成和对数据集的验证。
  - 调用时，注意在此脚本内协调好两个功能的日志，保证日志美观清晰。

### 3.3 模型训练、验证

- **FR-006：准备训练与验证配置文件（yolo/configs/train(val).yaml）**

  - 提供configs.py
  - 文件中应当提供 ultralytics 模型在训练、验证时允许设置的参数。
  - 涉及路径之处，全部使用相对于`yolo/`目录的相对目录，如 `yolo/utils/paths.py` 记作`utils/paths.py`，以此类推。

- **FR-007：参数处理工具（yolo/utils/config_utils.py）**

  - 提供参数读取函数 `load_yaml`

    - 函数参数：
      - config_name: str 配置文件名字。
      - logger: logging.Loger 日志记录器。
    - 函数返回值：
      - config_args: dict 配置文件中的参数。

  - 提供参数合并函数 `merge_args` 

    - 函数参数：

      - cli_args: Namespace 命令行参数的命名空间。

      - config_args: dict 配置文件中的参数。
      - mode: str 选取不同参数的组合的选项（"train" / "val"）。
      - logger: logging.Loger 日志记录器。

    - 函数返回值：

      - yolo_args: Namespace 仅包含YOLO官方允许的参数。
      - project_args: 包含所有参数。

  - 函数实现要求：

    - **如果认为函数功能需要切分，可自行切分为小函数，只要上述两个函数接口实现即可。**
    - yolo_args 在训练时需要传入 model.train方法，而project_args传入YOLO类的初始化方法，前者对参数审查严格，所以只能包含YOLO官方支持的参数。对这些参数可以手动初始化一个列表或一个文件便于确定yolo_args中应当包含和不包含的参数。
    - 在合并参数时，yaml文件中的相对路径需要标准化为绝对路径。
    - 在合并参数时，应当遵守 `CLI>YAML>Default` 的优先级。

- **FR-008：元信息工具（yolo/utils/meta_info_utils.py）**

  - 提供设备信息读取函数 `get_device_info`

    - 函数参数：
      - 无
    - 函数返回值
      - 系统及设备信息字典

  - 函数实现要求

    - 返回字典中应当包含下面的内容：

      - General

        | **字段**         | **描述**                                     |
        | ---------------- | -------------------------------------------- |
        | `OS_Type`        | 操作系统类型（如 Windows、Linux、Darwin）    |
        | `OS_Version`     | 操作系统版本号                               |
        | `Architecture`   | 系统架构（如 x86\_64、arm64）                |
        | `Python_Version` | 当前运行的 Python 版本号                     |
        | `System_Time`    | 当前系统时间（格式为 `YYYY-MM-DD HH:MM:SS`） |

      - CPU

        | **字段**           | **描述**                                            |
        | ------------------ | --------------------------------------------------- |
        | `Model`            | CPU 型号                                            |
        | `Cores`            | CPU 物理核心数                                      |
        | `Usage_Percentage` | 当前 CPU 使用率（百分比，格式化为带百分号的字符串） |

      - Memory

        | **字段**              | **描述**                                        |
        | --------------------- | ----------------------------------------------- |
        | `Total_RAM`           | 总内存大小（格式化为易读的单位，如 GB 或 MB）   |
        | `Available_RAM`       | 可用内存大小（格式化为易读的单位，如 GB 或 MB） |
        | `Used_RAM_Percentage` | 已使用内存的百分比（格式化为带百分号的字符串）  |

      - GPU

        | **字段**                | **描述**                                                     |
        | ----------------------- | ------------------------------------------------------------ |
        | `CUDA_Available`        | 是否支持 CUDA（布尔值）                                      |
        | `Pytorch_CUDA_Version`  | PyTorch 支持的 CUDA 版本（如果支持 CUDA）                    |
        | `Nvidia_Driver_Version` | Nvidia 驱动版本（如果支持 CUDA）                             |
        | `Device_Count`          | 可用的 GPU 设备数量（如果支持 CUDA）                         |
        | `Selected_Devices`      | 被选中的 GPU 设备数量（通过 PyTorch 的 `torch.cuda.device_count()` 获取） |
        | `Details`               | 每个 GPU 设备的详细信息（列表形式）                          |

        - 每个GPU的详细信息

          | **字段**           | **描述**                                            |
          | ------------------ | --------------------------------------------------- |
          | `Index`            | GPU 设备的索引号                                    |
          | `Model`            | GPU 型号（如 NVIDIA GeForce RTX 3080）              |
          | `Total_VRAM`       | GPU 的总显存大小（格式化为易读的单位，如 GB 或 MB） |
          | `CUDA_Capability`  | GPU 的 CUDA 计算能力版本                            |
          | `Usage_Percentage` | 当前 GPU 使用率（百分比，格式化为带百分号的字符串） |

      - Environment

        | **字段**               | **描述**                                      |
        | ---------------------- | --------------------------------------------- |
        | `Pytorch_Version`      | 当前安装的 PyTorch 版本号                     |
        | `ONNX_Runtime_Version` | ONNX Runtime 版本号（如果已安装）             |
        | `Cuda_Env`             | 当前环境中的 CUDA 版本号（通过 PyTorch 获取） |
        | `Cudnn_Version`        | 当前环境中的 cuDNN 版本号                     |
        | `Ultralytics_Version`  | Ultralytics YOLO 库的版本号                   |

  - 提供数据集元信息读取函数 `get_dataset_info`

    - 函数参数：

      - datayaml_path: Path 数据集配置文件路径

    - 函数返回值：

      - 数据集元信息字典

    - 函数实现要求

      - 返回字典应当包含以下内容

        | **字段**      | **描述**                           |
        | ------------- | ---------------------------------- |
        | `Config_File` | 配置文件路径                       |
        | `Class_Count` | 数据集中的类别数量                 |
        | `Class_Names` | 数据集中的类别名称列表             |
        | `Samples`     | 数据集中的样本数量                 |
        | `Data_Source` | 数据集的来源路径（训练集分割路径） |

  - 提供参数元信息读取函数 `get_args_info`

    - 函数参数：
      - args: Namespace
    - 函数返回值：
      - 参数元信息字典
    - 函数实现要求
      - 返回字典应当包含args中的所有内容

  - 提供运行结果元信息读取函数 `get_result_info`

    - 函数参数

      - result: ultralytics.utils.metrics.DetMetrics 训练或测试运行结果

    - 函数返回值

      - 运行结果元信息字典

    - 函数实现要求

      - 返回字典应当包含以下内容

        | **字段**      | **描述**               |
        | ------------- | ---------------------- |
        | **task**      | 任务类型               |
        | **save\_dir** | 保存训练结果的目录路径 |
        | **timestamp** | 时间戳（ISO 格式）     |

        - ProcessSpeed

          | **字段**        | **描述**                   |
          | --------------- | -------------------------- |
          | **preprocess**  | 预处理时间（单位：毫秒）   |
          | **inference**   | 推理时间（单位：毫秒）     |
          | **loss**        | 损失计算时间（单位：毫秒） |
          | **postprocess** | 后处理时间（单位：毫秒）   |

        - Metrics

          | **字段**      | **描述**                                 |
          | ------------- | ---------------------------------------- |
          | **Fitness**   | 模型的综合性能指标                       |
          | **Precision** | 精确率（预测为正的样本中实际为正的比例） |
          | **Recall**    | 召回率（实际为正的样本中预测为正的比例） |
          | **mAP50**     | 平均精度均值（IoU=0.5）                  |
          | **mAP50-95**  | 平均精度均值（IoU=0.5:0.95）             |

        - mAP50-90

          | **字段** | **描述**               |
          | -------- | ---------------------- |
          | 类别一   | 值为类别对应的mAP50-90 |
          | 类别二   | 值为类别对应的mAP50-90 |
          | ...      |                        |

- **FR-009：模型保存工具（yolo/utils/weights_utils.py）**
  - 提供权重文件复制函数 `copy_checkpoint`
    - 函数参数
      - source_dir: Path 权重文件原本所在目录
      - pretrained_name: str 模型原始预训练权重名称（如yolov8n）
      - logger: logging.Logger 日志记录器
    - 函数实现要求：
      - **权重文件复制后的名称应遵循格式：{train{id}}-{timestamp}-{pretrained_name}-{best/last}.pt**

- **FR-010：训练脚本（yolo/scripts/train.py）**

  - 本脚本的日志类型为 **train**

  - 本脚本的目的是调用上述功能实现对模型的训练。

  - 大致调用顺序如下

    1. 合并参数。
    2. 获取设备元信息，并调用`log_dict()`函数记录至日志。
    3. 获取数据集元信息，并调用`log_dict()`函数记录至日志。
    4. 获取参数信息，并调用`log_dict()`函数记录至日志。
    5. 执行训练。
    6. 获取运行结果信息，并调用`log_dict()`函数记录至日志。
    7. 复制模型到指定目录。
    8. 日志重命名（调用`rename_logfile`函数）。

  - 设置如下命令行参数（作为参考）

    ```python
    parser.add_argument("--data", type=str, default="data.yaml", help="path to data.yaml")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--device", type=str, default="0", help="device to use")
    parser.add_argument("--workers", type=int, default=8, help="number of workers")
    
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="path to weights.pt")
    ```

    

  - **此处仅列出实现训练脚本的基本框架，具体实现还应考虑异常处理、日志初始化等细节**

- **FR-011：验证脚本（yolo/scripts/val.py）**

  - 本脚本的日志类型为 **val**
  - 本脚本的目的是调用上述功能实现对模型的测试。
  - **此处默认使用参数split为 test**
  - 大致调用顺序如下
    1. 合并参数。
    2. 获取设备元信息，并调用`log_dict()`函数记录至日志。
    3. 获取数据集元信息，并调用`log_dict()`函数记录至日志。
    4. 获取参数信息，并调用`log_dict()`函数记录至日志。
    5. 执行测试。
    6. 获取运行结果信息，并调用`log_dict()`函数记录至日志。
    7. 日志重命名（调用`rename_logfile`函数）。

  - 设置如下命令行参数（作为参考）

    ```python
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--weights", type=str, default="train-20250613_164600-yolov8m-last.pt", help="Path to weights.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device to use")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    parser.add_argument("--split", type=str, default="test",choices = ["val", "test"], help="Split to use")
    ```

  - **此处仅列出实现验证脚本的基本框架，具体实现还应考虑异常处理、日志初始化等细节**

### 3.4 模型转换

- **FR-012：ONNX转换脚本（yolo/scripts/yolo2onnx.py）**
  - 本脚本的日志类型为 **yolo2onnx**
  - 本脚本的目的是将模型转化为ONNX格式，并保存在 `yolo/weights/onnx` 下。
  - 提供`yolo2onnx`函数，实现上述功能。函数详情在此不做规划。

### 3.5 应用程序

- **由于对pyside6技术栈并不熟悉，且app包内内容与其他部分耦合度较低，此处仅给出粗略功能需求，暂不做细致要求。**

- **FR-013：onnx模型移植脚本（app/cores/onnx_utils.py）**

  - 在本脚本中实现onnx模型文件移植。
  - 本脚本负责在项目启动时，将 `yolo/weights/onnx/` 目录下的模型拷贝到 `app/onnx/` 目录下 。
  - 如果某模型文件已经存在，则跳过拷贝。

- **FR-014：日志记录工具（app/cores/log_utils.py）**

  - 在本脚本中，实现日志记录函数，实现检测过程中的日志记录，每一条日志记录至少应包含以下数据：
    - 时间戳、数据源（图像名称/视频名称/摄像头编号）、目标在图像中的位置、置信度、类型、FPS等。

- **FR-015：模型推理脚本（app/cores/detect.py）**

  - 在本脚本，或者在onnx_utils.py中实现onnx模型的加载与推理功能。

- **FR-016：总体功能要求**

  - 应用程序至少应当实现以下功能
    1. 提供对图片、视频、实时摄像头等多种数据源的检测功能。
    2. 提供对火焰、烟雾目标的自动警示功能（声音、画面、远程通信）。
    3. 提供检测数据的UI界面展示功能（目标在图像中的位置、类型、置信度、FPS等）。
    4. 提供多个模型选项，并根据硬件配置自适应默认模型。
    5. 提供检测日志的实时记录（包含时间戳、数据源、目标数据等等信息）。

  - **其余功能及实现细节应当由前后端工程师根据技术栈具体情况详细规划**

### 3.6 测试脚本

- **FR-017 测试包（test/）**

  - **测试脚本应当对整个项目（YoloFire/）下的所有实现独立功能的脚本进行测试，确保这些脚本能够正常行使功能，并尽可能使项目代码的测试覆盖率达到100%。**

  - 预先设定的测试脚本目录结构如下，**可根据实际情况调整**：

    - 比如此处设置的目录完全针对于独立功能脚本，如有必要，也可以针对某些或全部工具脚本进行测试。

      ```bash
      test/
      ├── test_ui/
      │   ├── test_onnx_utils.py
      │   ├── test_log_utils.py
      │   ├── test_detect.py
      │   └── ... (其他测试文件)
      └── test_yolo/
          ├── test_initial_yolo.py
          ├── test_prepare_data.py
          ├── test_train.py
          ├── test_val.py
          └── test_yolo2onnx.py
      ```

## 4. 技术栈要求

- 统一python版本：3.12.4
- 统一pytorch版本：2.7.1

- 前后端工程师：
  - Pyside6
  - onnxruntime
- 测试工程师
  - unittest/pytest任选
  - coverage（用于统计测试覆盖率并生成测试报告）
  - pdb（不是包，用于调试）
  - objprint（可选）
  - viztracer（可选）

## 5. 验收标准

- **AC-001：项目初始化**：正确创建目录结构，日志记录操作。
- **AC-002：数据集转换**：生成 YOLO 格式标签、`data.yaml`，分割正确。
- **AC-003：模型训练**：生成 `best.pt`, `last.pt`，日志记录参数和结果。
- **AC-004：模型验证**：输出核心指标，保存结果到 `runs/val/`。
- **AC-005：模型推理**：支持多种输入，生成美化结果，触发语音提醒。
- **AC-006：日志质量**：日志文件按约定命名，包含设备、耗时、错误信息。
- **AC-007：代码质量**：符合 PEP 8，无 linting 警告。
- **AC-008：性能**：转换/验证 < 5 分钟，推理 < 0.1 秒/帧。
- **AC-009：文档**：README 和测试报告完整。

## 6. 外部参考文档

- Ultralytics YOLOv8/YOLOv11（https://docs.ultralytics.com/）。
- Python 文档（https://docs.python.org/3/）。