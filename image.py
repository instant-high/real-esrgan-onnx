import numpy as np
import cv2
from argparse import ArgumentParser

import onnxruntime as rt
rt.set_default_logger_severity(3)

parser = ArgumentParser()
parser.add_argument("--image", default='1.png', help="path to image")
parser.add_argument("--result", default='1_enh.png', help="path to result image")
opt = parser.parse_args()

from RealEsrganONNX.esrganONNX import RealESRGAN_ONNX
enhancer = RealESRGAN_ONNX(model_path="RealEsrganONNX/RealESRGAN_x2_fp16.onnx", device="cuda")
            
img = cv2.imread(opt.image)

#
height, width, channels = img.shape
if width %2 !=0:
    width = width - 1
if height %2 !=0:
    height = height - 1
img = cv2.resize(img,(width,height))
#

#use fp16 for faster inference
result = enhancer.enhance_fp16(img)
#result = enhancer.enhance(img)

cv2.imwrite(opt.result, result)
cv2.imshow("Result",result.astype(np.uint8))
cv2.waitKey() 
