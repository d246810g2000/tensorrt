"""trt_yolo.py

This script demonstrates how to do real-time face recognition with
TensorRT optimized YOLO engine and mobilefacenet engine.
"""

import os
import time
import argparse
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps

from utils.yolo import TRT_YOLO
from utils.face_recognition import TRT_MobileFacenet

WINDOW_NAME = 'Face_recognition'

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face recognition with TensorRT optimized '
            'YOLO model and mobilefacenet model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608)'))
    args = parser.parse_args()
    return args

def loop_and_detect(cam, trt_yolo, trt_mobilefacenet, conf_th):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      trt_mobilefacenet: the TRT mobilefacenet object extractor instance.
      conf_th: confidence/score threshold for object detection.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        facePositions, confs, clss = trt_yolo.detect(img, conf_th)
        embeddings = trt_mobilefacenet.face_encodings(img, facePositions)
        for facePosition, embedding in zip(facePositions, embeddings):
            name = 'Unknown Person' 
            matches = trt_mobilefacenet.compare_faces(embedding, threshold=1)
            distance = matches[1]
            if False not in matches:
                name = matches[0]
            x1, y1, x2, y2 = facePosition
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, name+', '+str(distance), (x1+2, y1+15), font, 0.6, (0, 255, 255), 2)

        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

    trt_yolo = TRT_YOLO(args.model, (h, w))
    trt_mobilefacenet = TRT_MobileFacenet(db_file='db.pkl')

    open_window(
        WINDOW_NAME, 'Camera TensorRT face recognition Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, trt_mobilefacenet, conf_th=0.5)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
