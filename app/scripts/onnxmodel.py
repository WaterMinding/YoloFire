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
        try:
            self.session = onnxrt.InferenceSession(
                model_path, 
                providers = providers
            )
        except Exception as e:
            raise e

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


    