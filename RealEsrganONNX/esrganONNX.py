import cv2
# import torch
import onnxruntime
import numpy as np


class RealESRGAN_ONNX:
    def __init__(self, model_path="RealESRGAN_x2.onnx", device='cuda'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        
    def enhance(self, img):
    
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = img /255
        img = np.expand_dims(img, axis=0).astype(np.float32)
        #
        result = self.session.run(None, {(self.session.get_inputs()[0].name):img})[0][0]
        #
        result = (result.squeeze().transpose((1,2,0)) * 255).clip(0, 255).astype(np.uint8)
        return result
    
    def enhance_fp16(self, img):
    
        img = img.astype(np.float16)
        img = img.transpose((2, 0, 1))
        img = img /255
        img = np.expand_dims(img, axis=0).astype(np.float16)
        #
        result = self.session.run(None, {(self.session.get_inputs()[0].name):img})[0][0]
        #
        result = (result.squeeze().transpose((1,2,0)) * 255).clip(0, 255).astype(np.uint8)
        return result
        
