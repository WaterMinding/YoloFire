# coding:utf-8
# @FileName: inference_main.py
# @Author  : BLC
# @Time    : 2025/6/19 14:32
# @Project : SafeH
# @Function: 推理代码，调用onnxmodel类中的函数对数据进行推理并返回结果
import cv2
import numpy as np
from pathlib import Path
from onnxmodel import ONNXModel  # 假设 ONNXModel 类在 onnxmodel.py 文件中


def preprocess_image(image, input_size):
    """预处理图像以适应模型输入尺寸"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)
    img = (np.expand_dims(img, 0) / 255.0).astype(np.float32)
    return img


def process_frame(frame, model, conf=0.25):
    """处理单帧图像"""
    img = preprocess_image(frame, model.input_shape)
    result = model.inference(img, conf)

    # 绘制检测结果
    for i in range(result.shape[0]):
        x1, y1, x2, y2, conf, clss = result[i]
        x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


def process_image(image_path, model):
    """处理图像文件并返回检测过后带框的图像"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像文件：{image_path}")
        return None

    result_img = process_frame(img, model)
    return result_img


def process_video(video_path, model):
    """处理视频文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件：{video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = process_frame(frame, model)
        cv2.imshow('Video', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_camera(model):
    """处理摄像头输入"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = process_frame(frame, model)
        cv2.imshow('Camera', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # 初始化模型
    model = ONNXModel(
        model_path=Path(r'/SafeUI/test.onnx'),
        input_size=(1, 3, 640, 640)
    )

    # 选择输入源
    input_source = 3

    if input_source == 1:
        image_path = input("请输入图像文件路径: ")
        process_image(image_path, model)
    elif input_source == 2:
        video_path = input("请输入视频文件路径: ")
        process_video(video_path, model)
    elif input_source == 3:
        process_camera(model)
    else:
        print("无效选择")


if __name__ == '__main__':
    main()