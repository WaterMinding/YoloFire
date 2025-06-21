# coding:utf-8
# @FileName: app.py
# @Author: BLC
# @Time: 2025/6/17 22:15
# @Project: SafeH
# @Function:
import os
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from pytts import init_tts
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from paths import LOGS_DIR
from paths import MODELS_DIR
from infer import FireInference
from logger import setup_logger
from yolofire_ui import Ui_MainWindow



def numpy_to_qpixmap(img_array):
    # 将 OpenCV 的 BGR 格式转换为 RGB 格式
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # 获取图像的高度、宽度和通道数
    height, width, channel = img_rgb.shape

    # 将 numpy 数组转换为 QImage
    # 参数说明：
    # img_rgb.data: 图像数据的字节数组
    # width: 图像宽度
    # height: 图像高度
    # width * channel: 每行的字节数
    # QImage.Format_RGB888: 指定图像格式为 RGB
    qimage = QImage(img_rgb.data, width, height, width * channel, QImage.Format.Format_RGB888)

    # 从 QImage 创建 QPixmap
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

class VideoThread(QThread):
    # 定义两个信号，分别用于原始帧和处理后的帧,通过emit方法发射信号给主线程并携带每一帧图像数据
    raw_pixmap_signal = pyqtSignal(QPixmap)
    processed_pixmap_signal = pyqtSignal(QPixmap, np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(self, model, confidence_threshold, logger):
        super().__init__()
        self.model = model  # 模型
        self.confidence_threshold = confidence_threshold  # 置信度
        self.logger = logger
        self.is_running = False
        self.show_processed = False  # 是否传输处理后的摄像头数据，用于控制检测是否开始

    def run(self):
        self.is_running = True
        # 打开摄像头或视频文件
        cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
        if not cap.isOpened():
            self.log_signal.emit("Failed to open camera")
            self.logger.error("打开摄像头失败")
            return
        cap.set(cv2.CAP_PROP_FPS, 60)  # 设置帧率为60fps，根据摄像头支持的最大帧率调整
        while self.is_running:
            ret, frame = cap.read()
            if ret:
                # 发送原始帧
                self.send_raw_frame(frame)

                # 调用 detect 函数进行目标检测处理
                processed_frame, result = self.detect(frame)
                if self.show_processed:
                    self.send_processed_frame(processed_frame, result)
            else:
                self.log_signal.emit("Failed to read frame from camera")
                self.logger.error("从摄像头读取数据失败")
                break
        cap.release()

    def send_raw_frame(self, frame):
        # # 将 BGR 格式转换为 RGB 格式
        # rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # h, w, ch = rgb_image.shape
        # bytes_per_line = ch * w
        # # 转换为 QImage 格式
        # convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # p = QPixmap.fromImage(convert_to_Qt_format)
        p = numpy_to_qpixmap(frame)
        p = p.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        self.raw_pixmap_signal.emit(p)  # 发送原始帧

    def send_processed_frame(self, frame, result):
        # # 将 BGR 格式转换为 RGB 格式
        # rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # h, w, ch = rgb_image.shape
        # bytes_per_line = ch * w
        # # 转换为 QImage 格式
        # convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # p = QPixmap.fromImage(convert_to_Qt_format)
        p = numpy_to_qpixmap(frame)
        p = p.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        self.processed_pixmap_signal.emit(p, result)  # 发送处理后的帧

    def detect(self, frame):
        p , result= self.model.process_video(frame, self.confidence_threshold)
        return p, result

    def stop(self):
        self.is_running = False
        self.logger.info("已停用摄像头检测线程")

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(1400, 900)
        self.video_cap = None  #
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)  # 计时器，每当时间超过设定期限就会调用connect的函数
        self.ui.belief_value.textChanged.connect(self.conf_changed)
        self.is_playing = True
        self.fps_value = self.ui.IOU_value  # 用于实时显示fps
        self.engine = init_tts()
        # 初始化标志位和线程
        self.hasflame = False
        self.hassmog = False
        self.alarm_thread = None
        self.alarm_lock = threading.Lock()
        self.is_alarm_active = False

        # 初始化模型菜单
        self.ui.select_model_path.addItems(["None"])

        model_names = list(MODELS_DIR.glob("*.onnx"))
        
        self.ui.select_model_path.addItems([
            model.stem for model in model_names
        ])

        self.ui.select_model_path.setCurrentText("None")

        self.ui.model_path.setText('None')

        # 初始化变量
        self.log_dir = LOGS_DIR
        self.model_path = None
        self.model = None
        self.data_path = ""  # 数据路径
        self.confidence_threshold = 0.7  # 置信度初值
        self.video_thread = None
        self.output_frame_list = []  # 保存视频文件输出的帧列表，用于将结果重新保存为视频文件
        self.ui.IOU.setText("帧率:")
        self.last_fps_time = time.time()  # 记录上一次计算FPS的时间
        self.fps = 0
        self.frame_count = 0
        self.logger = setup_logger(self.log_dir)
        # 设置初始值
        self.ui.data_path.setText(self.data_path)
        self.ui.belief_value.setText(f"{self.confidence_threshold:.2f}")
        self.fps_value.setText(f"{self.fps:.2f}")

        # 连接信号和槽
        self.ui.select_model_path.currentIndexChanged.connect(self.select_model)
        self.ui.select_data_path.clicked.connect(self.select_data)
        self.ui.start_button.clicked.connect(self.start_detection)
        self.ui.stop_button.clicked.connect(self.stop_detection)
        self.ui.end_button.clicked.connect(self.end_detection)
        self.ui.save_button.clicked.connect(self.save_results)
        self.ui.log_button.clicked.connect(self.open_log)
        self.ui.data_class.currentIndexChanged.connect(self.on_data_class_changed)

        # 设置默认文件类型过滤器
        self.file_type_filter = "Images (*.png *.xpm *.jpg);;Videos (*.mp4 *.avi *.mov)"

        self.logger.info(f"项目初始化完成，可以开始进行火灾检测")

    def conf_changed(self):
        self.confidence_threshold = float(self.ui.belief_value.text())
        self.logger.info(f"置信度已更改为{self.confidence_threshold:.2f}")

    def select_model(self):
        """
        选择模型路径
        :return:
        """
        model_stem = self.ui.select_model_path.currentText()

        if model_stem == 'None':

            self.model = None
            self.model_path = None
            self.ui.model_path.setText("None")

            self.logger.info(f"未选择模型")
        
        else:

            model_path = Path(
                MODELS_DIR / (model_stem + ".onnx")
            )
            self.model_path = str(model_path.relative_to(Path.cwd()))

        if self.model_path:
            self.model = FireInference(self.model_path)
            self.ui.model_path.setText(str(model_path))
            self.logger.info(f"模型路径已设置为：{self.model_path}")


    def select_data(self):
        """
        选择数据路径
        :return:
        """
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先选择模型文件。")
            self.logger.warning("未选择模型文件！请先选择模型文件再选择数据路径")
            return
        options = QFileDialog.Option.DontUseNativeDialog
        if self.ui.data_class.currentText() == "图片":
            self.file_type_filter = "Images (*.png *.xpm *.jpg)"
        elif self.ui.data_class.currentText() == "视频":
            self.file_type_filter = "Videos (*.mp4 *.avi *.mov)"

        self.data_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", self.file_type_filter, options=options)
        if self.data_path:
            self.ui.data_path.setText(self.data_path)
            self.logger.info(f"数据路径已设置为：{self.data_path}")
            self.display_data()

    def on_data_class_changed(self, index):
        """
        每次数据类型更改时调用，检测是否改为了摄像头
        :param index:
        :return:
        """
        if self.ui.data_class.currentText() == "摄像头":
            self.ui.select_data_path.setEnabled(False)
            self.logger.info("数据类型已更改为摄像头")
            # self.open_camera()
        else:
            self.ui.select_data_path.setEnabled(True)
            if self.ui.data_class.currentText() == "视频":
                self.logger.info("数据路径已更改为视频")
            else:
                self.logger.info("数据路径已更改为图片")
    def open_camera(self):
        """
        打开摄像头
        :return:
        """
        self.video_thread = VideoThread(self.model, self.confidence_threshold, self.logger)
        self.video_thread.raw_pixmap_signal.connect(self.update_raw_image)
        self.video_thread.processed_pixmap_signal.connect(self.update_processed_image)
        self.video_thread.start()
        self.logger.info(f"已启用摄像头检测线程")


    def update_raw_image(self, image):
        """
        传输未经处理的摄像头画面
        :param image:
        :return:
        """
        self.ui.source_data.setPixmap(image)

    def update_processed_image(self, image, result):
        """
        传输处理过的摄像头画面
        :param result:
        :param image:
        :return:
        """
        self.update_information(result)
        self.hasflame = False
        self.hassmog = False
        for row in result:
            clss = int(row[-1])
            if clss == 0:
                self.hasflame = True
            else:
                self.hassmog = True

        self.manage_alarms()
        self.ui.result_data.setPixmap(image)
        np_processed_p = self.qPixmap_to_numpy(image)
        self.output_frame_list.append(np_processed_p)
        self.calculate_fps()

    def display_data(self):
        """
        展示原始数据(图片和视频)
        :return:
        """
        if self.data_path:
            if self.ui.data_class.currentText() == "图片":
                # 显示图片
                pixmap = QPixmap(self.data_path)
                scaled_pixmap = pixmap.scaled(self.ui.source_data.size(), Qt.AspectRatioMode.KeepAspectRatio)
                self.ui.source_data.setPixmap(scaled_pixmap)
            elif self.ui.data_class.currentText() == "视频":

                # 初始化视频播放器
                self.video_cap = cv2.VideoCapture(self.data_path)
                if not self.video_cap.isOpened():
                    QMessageBox.warning(self, "警告", "无法打开视频文件。")
                    self.logger.error("打开视频文件失败")
                    return
                # 显示第一帧
                ret, frame = self.video_cap.read()
                if ret:
                    self.show_frame(frame)

    def show_frame(self, frame):
        """
        展示视频数据每一帧画面
        :param frame:
        :return:
        """
        p = numpy_to_qpixmap(frame)
        self.ui.source_data.setPixmap(p)

    def show_processed_frame(self, frame):
        """
        展示处理过后的视频数据每一帧画面
        :param frame:
        :return:
        """
        p, result = self.model.process_video(frame, self.confidence_threshold)
        processed_p = numpy_to_qpixmap(p)
        np_processed_p = self.qPixmap_to_numpy(processed_p)
        self.output_frame_list.append(np_processed_p)
        self.update_information(result)
        self.hasflame = False
        self.hassmog = False
        for row in result:
            clss = int(row[-1])
            if clss == 0:
                self.hasflame = True
            else:
                self.hassmog = True

        self.manage_alarms()
        self.ui.result_data.setPixmap(processed_p)


    def update_video_frame(self):
        """
        更新视频数据每一帧画面,随timer重复频繁调用实现视频流畅播放
        :return:
        """
        if self.video_cap and self.is_playing:
            ret, frame = self.video_cap.read()
            if ret:
                self.show_frame(frame)
                self.show_processed_frame(frame)
                self.calculate_fps()
            else:
                self.video_timer.stop()
                self.video_cap.release()
                self.video_cap = None
                self.is_playing = False

    def stop_video(self):
        """
        终止视频检测
        :return:
        """
        if self.video_cap:
            self.video_timer.stop()
            self.logger.info(f"已终止视频检测")
            self.video_cap.release()
            self.video_cap = None
            self.is_playing = True

    def stop_camera(self):
        """
        终止摄像头检测
        :return:
        """
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            # 断开信号连接
            self.video_thread.raw_pixmap_signal.disconnect(self.update_raw_image)
            self.video_thread.processed_pixmap_signal.disconnect(self.update_processed_image)
            self.video_thread = None
            self.logger.info(f"已终止摄像头检测")
        # 清除摄像头画面
        self.ui.source_data.clear()
        self.ui.result_data.clear()


    def start_detection(self):
        """
        开始检测算法
        :return:
        """
        # if self.ui.data_class.currentText() == "摄像头":
        #     if not self.model_path:
        #         QMessageBox.warning(self, "警告", "请先选择模型文件")
        #         return
        #
        # elif not self.model_path or not self.data_path:
        #     QMessageBox.warning(self, "警告", "请先选择模型文件和数据文件。")
        #     return
        if self.ui.data_class.currentText() == "摄像头":
            if self.video_thread:
                self.video_thread.show_processed = True
                self.logger.info(f"继续摄像头检测")
            else:
                self.logger.info(f"开始摄像头检测")
                self.open_camera()
                self.output_frame_list.clear()
                # 清除之前的信息
                self.ui.text_information.clear()
                self.video_thread.show_processed = True
        elif self.ui.data_class.currentText() == "视频":
            if self.video_cap and self.is_playing:
                # 新开始一个检测
                self.logger.info("开始视频检测")
                self.output_frame_list.clear()
                # 清除之前的信息
                self.ui.text_information.clear()
                self.video_timer.start(10)  # 约100帧每秒
            if self.video_cap and not self.is_playing:
                self.is_playing = True
                self.logger.info(f"继续视频检测")
                self.video_timer.start(10)  # 约100帧每秒
        else:  # 检测图片
            # 清除之前的信息
            self.logger.info(f"开始图片检测")
            self.ui.text_information.clear()
            processed_img_ndarray, result = self.model.process_image(self.data_path, self.confidence_threshold)
            processed_img = numpy_to_qpixmap(processed_img_ndarray)
            scaled_processed_img = processed_img.scaled(self.ui.result_data.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.update_information(result)
            self.hasflame = False
            self.hassmog = False
            for row in result:
                clss = int(row[-1])
                if clss == 0:
                    self.hasflame = True
                else:
                    self.hassmog = True

            self.manage_alarms()
            self.ui.result_data.setPixmap(scaled_processed_img)

    def stop_detection(self):
        # 中止检测
        if self.ui.data_class.currentText() == "摄像头":
            self.video_thread.show_processed = False
            self.logger.info(f"已中止摄像头检测")
        elif self.ui.data_class.currentText() == "视频":
            if self.video_cap and self.is_playing:
                self.is_playing = False
                self.video_timer.stop()
                self.logger.info(f"已中止视频检测")
        else:
            pass

    def end_detection(self):
        # 终止检测
        if self.ui.data_class.currentText() == "摄像头":
            self.stop_camera()

        elif self.ui.data_class.currentText() == "视频":
            self.stop_video()
        else:
            self.logger.info("已终止图片检测")
        self.hasflame = False
        self.hassmog = False
        self.ui.source_data.clear()
        self.ui.result_data.clear()
        self.ui.data_path.clear()
        self.logger.info(f"本次检测日志文件保存在：logs 文件夹下")

    def save_results(self):
        """保存检测结果"""
        if self.ui.data_class.currentText() == "图片":
            # 保存处理后的图片
            if self.ui.result_data.pixmap() is None:
                QMessageBox.warning(self, "警告", "没有检测结果可以保存。")
                self.logger.error("没有检测结果可以保存，请先开启一次检测")
                return
            options = QFileDialog.Option.DontUseNativeDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存结果图片", "", "Images (*.png *.xpm *.jpg)", options=options)
            if file_path:
                # 确保文件有扩展名
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.xpm')):
                    file_path += '.png'  # 默认使用 PNG 格式
                result_pixmap = self.ui.result_data.pixmap()
                result_pixmap.save(file_path)
                QMessageBox.information(self, "信息", "结果已保存到: {}".format(file_path))
                self.logger.info(f"检测结果已保存至：{file_path}")

        elif self.ui.data_class.currentText() == "视频":
            self.end_detection()
            self.save_frames_to_video(self.output_frame_list)

        elif self.ui.data_class.currentText() == "摄像头":
            self.end_detection()
            self.save_frames_to_video(self.output_frame_list)

    def qPixmap_to_numpy(self, qPixmap):
        # 将 QPixmap 转换为 QImage
        qImage = qPixmap.toImage()

        # 确保 QImage 格式为格式为 ARGB32
        buffer = qImage.bits().asstring(qImage.width() * qImage.height() * 4)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape((qImage.height(), qImage.width(), 4))

        # 转换为 RGB 格式，因为 OpenCV 使用 BGR 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 如果需要，可以进一步转换为灰度图像
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image
    def save_frames_to_video(self, processed_frames, fps=30):
        """
        将处理后的帧保存为视频文件

        :param processed_frames: 处理后的帧列表
        :param fps: 帧率，默认为30
        """
        if not processed_frames:
            QMessageBox.warning(self, "警告", "没有可保存的帧。")
            return

            # 打开文件对话框，让用户选择保存路径和文件名
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存视频文件", "", "视频文件 (*.mp4 *.avi);;所有文件 (*)", options=options)

        if file_path:
            # 确保文件有扩展名
            if not file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                file_path += '.mp4'  # 默认使用 MP4 格式

            # 获取帧的分辨率
            frame_size = (processed_frames[0].shape[1], processed_frames[0].shape[0])

            # 定义视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID' 用于 AVI

            # 初始化视频写入器
            out = cv2.VideoWriter(file_path, fourcc, fps, frame_size)

            # 逐帧写入视频文件
            for frame in processed_frames:
                out.write(frame)

            # 释放资源
            out.release()
            QMessageBox.information(self, "信息", f"视频已保存到: {file_path}")
            self.logger.info(f"检测结果已保存至：{file_path}")

    def update_information(self, result):
        """更新检测信息到 QPlainTextEdit"""
        # 清除之前的信息（如果需要）
        # self.text_information.clear()

        # 定义类别名称
        class_names = {0: "frame", 1: "smog"}

        # 遍历检测结果并更新信息
        for detection in result:
            x1, y1, x2, y2, score, class_id = detection
            class_name = class_names.get(int(class_id), "unknown")
            info = f"检测到类别: {class_name}, 置信度: {score:.2f}, 位置: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f})"
            self.ui.text_information.appendPlainText(info)
            self.logger.info(info)

        # 滚动到最新信息
        self.ui.text_information.ensureCursorVisible()

    def manage_alarms(self):
        # 只有在没有活跃的报警时，才处理新的报警
        if not self.is_alarm_active:
            if self.hasflame:
                self.is_alarm_active = True
                self.alarm_thread = threading.Thread(
                    target = self.alarm_frame_tts,
                    daemon = True
                )
                self.alarm_thread.start()
            elif self.hassmog and not self.hasflame:
                self.is_alarm_active = True
                self.alarm_thread = threading.Thread(
                    target = self.alarm_smog_tts,
                    daemon = True
                )
                self.alarm_thread.start()



    def alarm_frame_tts(self):
        while self.hasflame and self.is_alarm_active:
            with self.alarm_lock:
                self.engine.say("警告!警告！目标检测区域有明火产生，请消防人员快速前往查看。")
                self.engine.runAndWait()
        self.is_alarm_active = False

    def alarm_smog_tts(self):
        while self.hassmog and not self.hasflame and self.is_alarm_active:
            with self.alarm_lock:
                self.engine.say("警告!警告！目标检测区域有大量烟雾，可能有消防隐患，请消防人员快速前往查看。")
                self.engine.runAndWait()
        self.is_alarm_active = False

    def open_log(self):
        self.open_log_file(self.log_dir)

    def open_log_file(self, log_dir, encoding="utf-8"):
        """
            打开日志文件
            :param log_dir: 日志文件所在的目录
            :param encoding: 文件的编码格式
            """
        # 列出目录中的所有日志文件
        log_files = [f for f in os.listdir(log_dir) if f.startswith("app_") and f.endswith(".log")]

        if not log_files:
            print("No log files found in the specified directory.")
            return

        # 假设我们读取最新的日志文件
        log_files.sort(reverse=True)  # 按文件名排序，最新的文件在前
        latest_log_file = log_files[0]
        log_file_path = os.path.join(log_dir, latest_log_file)

        # 打开日志文件
        os.startfile(log_file_path)

    def calculate_fps(self):
        """
        计算实时帧率（FPS）
        """
        self.frame_count += 1
        current_time = time.time()

        # 每隔1秒计算一次FPS
        if current_time - self.last_fps_time >= 1:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            self.fps_value.setText(f"{self.fps:.2f}")
    
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
