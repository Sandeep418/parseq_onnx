
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
# state_dict = torch.load(checkp, map_location=torch.device('cpu'))
# parseq.load_state_dict(state_dict, strict=False)  # Usparseq.eval()
parseq = load_from_checkpoint(checkp).eval().to('cpu')
#parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()#torch.load(checkp).eval()#
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


save_dir=r'D:\OCR_DATA\Inference_op/'

img_dir=r"D:\OCR_DATA\Benchmark_A1/"


text_file_dir=r'D:\OCR_DATA\ocr_text_files/A1/'
#r'C:\Users\Sandeep.Reddy\Downloads\NKP_data\No_pharma_set_1\line_crops_Image_processing/'
for img_name in os.listdir(img_dir):
    image_path=img_dir+img_name
    fp=open(text_file_dir+img_name[:-4]+'.txt','w')

    inpimg = Image.open(image_path).convert('RGB')

    img_data = np.array(inpimg)
    # Preprocess. Model expects a batch of images with shape: (B, C, H, W)
    img = img_transform(inpimg).unsqueeze(0)
    print("---------------------Image Name is :", img_name, "-------------------------------")
    st = time()
    logits = parseq(img)
    print("Time Taken for the Inference is:",time()-st) # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

    # Greedy decoding
    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    #print("Inference Time is :",time()-st)
    print('Decoded label = {}'.format(label[0]))
    print("confidence :",confidence)

    save_img=np.array(inpimg)
    ht, wd, cc = save_img.shape
    ww = save_img.shape[1]
    hh = save_img.shape[0]
    color = (255, 255, 255)
    result = np.full((hh, ww, 3), color, dtype=np.uint8)

    # set offsets for top left corner
    xx = 0
    yy = 0

    thickness=1
    fontScale = 0.8

    #fp.write(str(label[0]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.putText(result, str(str(label[0])), (0,ht//2), font,
                        fontScale, (0,255,0), thickness, cv2.LINE_AA)


    # result = cv2.putText(result, str(confidence), (0, ht-5), font,
    #                      0.3, (0, 0, 255), thickness, cv2.LINE_AA)

    vis = np.concatenate((save_img, result), axis=1)
    print("******************************************************************************")
    cv2.imshow("cresult",vis)
    cv2.imshow("img", np.array(inpimg))
    cv2.imwrite(save_dir+img_name,vis)
    fp.close()
    cv2.waitKey(0)


