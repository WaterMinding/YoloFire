from pathlib import Path

import importlib
import numpy as np
from numpy import ndarray
import onnxruntime as onnxrt



class ONNXModel:

    def __init__(
        self,
        model_path: Path, 
        input_size: tuple[int, int, int, int],
        use_trt: bool = True
    ):
        
        self.model_path = model_path

        self.input_shape = np.array(input_size)

        providers = [
            'CUDAExecutionProvider', 
            'CPUExecutionProvider'
        ]

        if use_trt:

            importlib.import_module('tensorrt')

            trt_ep_options = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": 
                    str(model_path.parent / 'trt_cache'),
            }

            providers.insert(       
                0, ('TensorrtExecutionProvider', trt_ep_options)
            )
        
        # print('providers: ', providers)

        self.session = onnxrt.InferenceSession(
            model_path, 
            providers = providers
        )

    # 推理方法
    def inference(self, inp: ndarray, conf: float = 0.25) -> ndarray:

        input_name = self.session.get_inputs()[0].name

        outputs = self.session.run(
            None, {input_name: inp}
        )

        out_array = outputs[0].reshape(-1, 6)

        anchors =  out_array[
            ~np.all((out_array==0), axis = 1)
        ]

        value_mask = np.array([
            self.input_shape[3],
            self.input_shape[2],
            self.input_shape[3],
            self.input_shape[2],
            1, 1
        ])

        anchors = anchors / value_mask
        conf_mask = anchors[:, 4] > conf

        anchors = anchors[conf_mask]

        return anchors


if __name__ == '__main__':

    import time


    start = time.time()
    model = ONNXModel(
        model_path = Path(r'/SafeUI/test.onnx'),
        input_size = (1, 3, 640, 640),
        use_trt = True,
    )
    print('load time:', time.time() - start)

    import cv2

    imgx = cv2.imread(r'D:\SafeH\fire_data\VCG211519899462.jpg')


    img = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (640, 640))

    img = img.transpose(2, 0, 1)

    img = (np.expand_dims(img, 0) / 255.0).astype(np.float32)

    
    for i in range(1):
        start = time.time()
        result = model.inference(img)
        print(time.time() - start)
    
    # 绘制结果
    for i in range(result.shape[0]):
        x1, y1, x2, y2, conf, clss = result[i]
        x1, y1, x2, y2 = int(x1 * imgx.shape[1]), int(y1 * imgx.shape[0]), int(x2 * imgx.shape[1]), int(y2 * imgx.shape[0])
        cv2.rectangle(imgx, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('img', imgx)
    cv2.waitKey(0)
    