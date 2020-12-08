import common
import cv2
import time
import numpy as np
import tensorrt as trt

# 只報告警告和錯誤
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_path):
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def main():
    # load label
    labels = [line.rstrip('\n') for line in open('class_labels.txt')]

    # load engine
    trt_engine = './vgg16_32.trt'
    Cifar10_engine = load_engine(trt_engine)

    dispW = 1280
    dispH = 720
    flip = 0
    fpsReport = 0
    camSet = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=60/1 \
              ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+',\
               format=BGRx ! videoconvert !video/x-raw, format=BGR ! appsink'
    cap = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)
    timeStamp = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        _, frame = cap.read()
        frame = frame.astype('float32')
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = cv2.resize(frameRGB, (32, 32))/255
        img = img.transpose((2, 0, 1)).flatten()

        # 分配buffers給inputs和outputs
        inputs, outputs, bindings, stream = common.allocate_buffers(Cifar10_engine)
        inputs[0].host = img

        # inference
        with Cifar10_engine.create_execution_context() as context:
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = trt_outputs[0].argmax(-1)

        # fps
        dt = time.time() - timeStamp
        fps = 1/dt 
        fpsReport = .9*fpsReport + .1*fps
        timeStamp = time.time()
        cv2.rectangle(frame, (0, 0), (350+len(labels[pred])*30, 80), (0, 0, 255), -1)
        cv2.putText(frame, str(round(fpsReport, 1))+'fps'+', '+labels[pred], (0, 60), font, 2, (0, 255, 255), 3)
 
        cv2.imshow('stream', frame/255)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
