import cv2
import os
import time
import torch
import common
import argparse
import numpy as np
import tensorrt as trt
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default='image', help='demo type, eg. image, video, webcam and csicam')
    parser.add_argument('--config', help='model config file path')
    parser.add_argument('--model', help='model file path')
    parser.add_argument('--path', default='./demo', help='path to images or video')
    parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
    args = parser.parse_args()
    return args


class Predictor(object):

    def _load_engine(self):
        TRTbin = self.trt_model
        with open(TRTbin, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    def __init__(self, cfg, model_path, logger, device='cuda:0'):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        
        self.trt_model = model_path
        self.inference_fn = common.do_inference if trt.__version__[0] < '7' \
                                                else common.do_inference_v2
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()
        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                common.allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def __del__(self):
        del self.outputs
        del self.inputs
        del self.stream

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = np.expand_dims(meta['img'].transpose(2, 0, 1), axis=0)
        self.inputs[0].host = meta['img'].flatten()
        time1 = time.time()
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        time2 = time.time()
        print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
        output_shapes = [(1, 80, 40, 40), (1, 32, 40, 40), (1, 80, 20, 20), (1, 32, 20, 20), (1, 80, 10, 10), (1, 32, 10, 10)]
        cls_score1, bbox_pred1, cls_score2, bbox_pred2, cls_score3, bbox_pred3 = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)] 
        
        cls_scores = [torch.from_numpy(cls_score1).cuda(0), torch.from_numpy(cls_score2).cuda(0), torch.from_numpy(cls_score3).cuda(0)]
        bbox_preds = [torch.from_numpy(bbox_pred1).cuda(0), torch.from_numpy(bbox_pred2).cuda(0), torch.from_numpy(bbox_pred3).cuda(0)]
        results = self.model.head.post_process((cls_scores, bbox_preds), meta)
        print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=True)
        print('viz time: {:.3f}s'.format(time.time()-time1))


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



def main():
    args = parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(-1, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device='cuda:0')
    logger.log('Press "Esc", "q" or "Q" to exit.')
    fpsReport = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    timeStamp = time.time()
    if args.demo == 'image':
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            predictor.visualize(res, meta, cfg.class_names, 0.35)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
    elif args.demo == 'video' or args.demo == 'webcam':
        cap = cv2.VideoCapture(args.path if args.demo == 'video' else args.camid)
        while True:
            ret_val, frame = cap.read()
            meta, res = predictor.inference(frame)
            predictor.visualize(res, meta, cfg.class_names, 0.35)
            dt = time.time() - timeStamp
            fps = 1/dt 
            fpsReport = .9*fpsReport + .1*fps
            print("FPS: "+ str(fpsReport))
            timeStamp = time.time()
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
    elif args.demo == 'csicam':
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        while True:
            ret_val, frame = cap.read()
            meta, res = predictor.inference(frame)
            predictor.visualize(res, meta, cfg.class_names, 0.35)
            dt = time.time() - timeStamp
            fps = 1/dt 
            fpsReport = .9*fpsReport + .1*fps
            print("FPS: "+ str(fpsReport))
            timeStamp = time.time()
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break        


if __name__ == '__main__':
    main()
