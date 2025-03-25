
import os

import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
import cv2
import numpy as np
from time import time
import onnx
# Load model and image transforms

# parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).to('cpu')#PARSeq.from_pretrained("parseq_base")
parseq = load_from_checkpoint(checkp).eval().to('cpu')
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

parseq.refine_iters = 0
parseq.decode_ar = False
batch_size = 1
sequence_length = 25  # Adjust as per your dataset
batch_size = 1
channels = 3  # RGB
height = 32   # Adjust based on model training
width = 128   # Adjust based on model training

dummy_input = torch.randn(batch_size, channels, height, width)  # Correct shape

# Define ONNX export path
onnx_path = "parseq_trained_12.onnx"

# Export the model
torch.onnx.export(
    parseq,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17
)

# Load and check the ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print("PARSeq model successfully converted to ONNX and verified!")

