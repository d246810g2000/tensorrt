import cv2
import common
import numpy as np
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_labels(label_file_path):
    labels = [line.rstrip('\n') for line in open(label_file_path)]
    return labels

def preprocess(img_path, img_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    img = np.float32(img)/255.0
    img = cv2.resize(img, (img_size, img_size))
    return img

# 讀進照片並進行前處理
img_path = 'tabby_tiger_cat.jpg'
img_size = 224
img = preprocess(img_path, img_size)

# 獲得 labels 的資訊
label_file_path = 'class_labels.txt'
labels = load_labels(label_file_path)

# load trt engine
trt_path = 'mobilenet.trt'
with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    
# 分配 buffers 給 inputs 和 outputs
inputs, outputs, bindings, stream = common.allocate_buffers(engine)
inputs[0].host = img

# inference
with engine.create_execution_context() as context:
    trt_outputs = common.do_inference_v2(context, 
                                         bindings=bindings, 
                                         inputs=inputs, 
                                         outputs=outputs, 
                                         stream=stream)
# 預測結果
idx = trt_outputs[0].argmax(-1)
print(f'Predicted: {labels[idx]}, {trt_outputs[0][idx]}')