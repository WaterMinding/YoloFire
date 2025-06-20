# coding:utf-8
# @FileName: infer.py
# @Author: BLC
# @Time: 2025/6/19 15:15
# @Project: SafeH
# @Function:
import cv2
import numpy as np
from pathlib import Path
from onnxmodel import ONNXModel  # 假设 ONNXModel 类在 onnxmodel.py 文件中


class FireInference:
    def __init__(self, model_path, input_size=(1, 3, 640, 640)):
        """初始化 FireInference 类"""
        self.model = ONNXModel(
            model_path=Path(model_path),
            input_size=input_size

        )
    def paint_rectangle(self, img_path, array):
        """
        画框函数，用于在原始图像上画出检测到的框
        :param img_path: 原始图像路径
        :param array: 数据框位置数组
        :return: 经过画框处理的图像文件
        """


    def preprocess_image(self, image):
        """预处理图像以适应模型输入尺寸"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose(2, 0, 1)
        img = (np.expand_dims(img, 0) / 255.0).astype(np.float32)
        return img

    def process_frame(self, frame, conf=0.25):
        """处理单帧图像"""
        img = self.preprocess_image(frame)
        result = self.model.inference(img, conf)

        # 提取框、置信度和类别
        boxes = []
        confs = []
        labels = []
        for i in range(result.shape[0]):
            x1, y1, x2, y2, conf, clss = result[i]
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(
                y2 * frame.shape[0])
            boxes.append([x1, y1, x2, y2])
            confs.append(conf)
            labels.append(clss)

        # 自定义标签和颜色映射
        custom_label_mapping = {
            0: "flame",
            1: "smog"
        }

        custom_color_mapping = {
            0: (0, 0, 255),  # 红色 (BGR)
            1: (0, 255, 255)  # 黄色 (BGR)
        }

        # 调用美化函数
        from beautify import custom_plot
        beautified_frame = custom_plot(
            frame,
            boxes=boxes,
            confs=confs,
            labels=labels,
            use_chinese_mapping=False,  # 使用英文标签
            font_path=r"/LXGWWenKai-Bold.ttf",
            font_size=20,
            line_width=4,
            label_padding_x=10,
            label_padding_y=10,
            radius=8,
            text_color_bgr=(0, 0, 0),
            LABEL_MAPPING=custom_label_mapping,
            COLOR_MAPPING=custom_color_mapping
        )

        return beautified_frame, result




    def process_image(self, image_path, conf):
        """处理图像文件并返回检测过后带框的图像"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像文件：{image_path}")
            return None

        result_img, result = self.process_frame(img, conf)
        return result_img, result

    def process_video(self, frame, conf):
        # """处理视频文件"""
        # cap = cv2.VideoCapture(video_path)
        # if not cap.isOpened():
        #     print(f"无法打开视频文件：{video_path}")
        #     return
        #
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        result_frame, result = self.process_frame(frame, conf)
        return result_frame, result
