"""trt_face_recognition.py

This script demonstrates how to do real-time face recognition with
TensorRT optimized mobilefacenet engine.
"""

import os
import cv2
import time
import argparse
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.face_detection import TRT_RetinaFace
from utils.face_recognition import TRT_MobileFacenet
from utils.prior_box import PriorBox
from data import cfg_mnet

WINDOW_NAME = 'Face_recognition'

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face recognition with TensorRT optimized '
            'retinaface model and mobilefacenet model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[retinaface]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 320, 640)'))
    args = parser.parse_args()
    return args

def loop_and_detect(cam, trt_retinaface, trt_mobilefacenet, priors, cfg):
    """Continuously capture images from camera and do face recognition.

    # Arguments
      cam: the camera instance (video source).
      trt_retinaface: the TRT_RetinaFace face detector instance.
      trt_mobilefacenet: the TRT mobilefacenet face extractor instance.
      priors: priors boxes with retinaface model
      cfg: retinaface model parameter configure
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
        facePositions, landmarks = trt_retinaface.detect(priors, cfg, img)
        embeddings = trt_mobilefacenet.face_encodings(img, landmarks)
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
    if not os.path.isfile('retinaface/%s.trt' % args.model):
        raise SystemExit('ERROR: file (retinaface/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cfg = cfg_mnet
    input_size = args.model.split('-')[-1]
    input_shape = (int(input_size), int(input_size))
    priorbox = PriorBox(cfg, input_shape)
    priors = priorbox.forward()
    trt_retinaface = TRT_RetinaFace(args.model, input_shape)
    trt_mobilefacenet = TRT_MobileFacenet(db_file='db.pkl')

    open_window(
        WINDOW_NAME, 'Camera TensorRT Face Recognition Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_retinaface, trt_mobilefacenet, priors, cfg)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
