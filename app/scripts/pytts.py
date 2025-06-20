# coding:utf-8
# @FileName: pytts.py
# @Author: BLC
# @Time: 2025/6/16 15:20
# @Project: SafeH
# @Function: 语音功能
import pyttsx3
import time

def init_tts():
    """
    初始化语音合成引擎
    :return:
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 130)  # 设置语速
        engine.setProperty('volume', 1.0)  # 音量大小
        # 尝试使用中文语音包
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'zh' in voice.id.lower() or "chinese" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        return engine
    except Exception as e:
        print(f"初始化语音合成引擎失败：{e}")
        return None
def test_tts():
    """测试语音输出"""
    engine = init_tts()
    if not engine:
        print("初始化语音合成引擎失败")
    text = "警告!警告！目标检测区域有明火产生，请消防人员快速前往查看。"
    print(f"正在播放：{text}")
    try:
        engine.say(text)
        engine.runAndWait()
        print("播放完成")
    except Exception as e:
        print(f"播放失败:{e}")
    time.sleep(1)

if __name__ == "__main__":
    test_tts()