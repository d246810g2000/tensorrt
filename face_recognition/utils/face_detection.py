"""face_detection.py

Implementation of trt_face_detection
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from utils.common import *
from utils.box_utils import decode, decode_landm, py_cpu_nms

def _preprocess(img, input_shape):
    """Preprocess an image before TRT retinaface inferencing.

    # Args
        img: uint8 numpy array of shape (img_h, img_w, 3)
        input_size: model input size

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    img = np.float32(img)
    img -= (104, 117, 123)
    height, width, _ = img.shape
    long_side = max(height, width)
    img_pad = np.zeros((long_side, long_side, 3), dtype=img.dtype)
    img_pad[0:0+height, 0:0+width] = img
    img = cv2.resize(img_pad, input_shape)
    img = img.transpose((2, 0, 1))
    return img

def _postprocess(loc, conf, landms, priors, cfg, img):
    """Postprocess TensorRT outputs.

    # Args
        loc: [x, y, w, h]
        conf: [not object confidence, object confidence]
        landms: [eye_left.x, eye_left.y, 
                 eye_right.x, eye_right.y,
                 nose.x, nose.y
                 mouth_left.x, mouth_right.y
                 mouth_left.x, mouth_right.y]
        priors: priors boxes with retinaface model
        cfg: retinaface model parameter configure
        img: input image

    # Returns
        facePositions, landmarks (after NMS)
    """
    long_side = max(img.shape)
    img_size = cfg['image_size']
    variance = cfg['variance']
    scale = np.ones(4)*img_size
    scale1 = np.ones(10)*img_size
    confidence_threshold = 0.2
    top_k = 50
    nms_threshold = 0.5

    # decode boxes
    boxes = decode(np.squeeze(loc, axis=0), priors, variance)
    boxes = boxes*scale

    # decode landmarks
    landms = decode_landm(np.squeeze(landms, axis=0), priors, variance)
    landms = landms*scale1  

    # ignore low scores
    scores = np.squeeze(conf, axis=0)[:, 1]
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]   

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS 
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]
    
    # resize
    res = long_side/img_size
    facePositions = (dets[:,:4]*res).astype(int).tolist()
    landmarks = (landms*res).astype(int).tolist()
    return facePositions, landmarks

class TRT_RetinaFace(object):
    """TRT_RetinaFace class encapsulates things needed to run TRT detection."""

    def _load_engine(self):
        TRTbin = 'retinaface/%s.trt' % self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, model, input_shape, cuda_ctx=None):
        """Initialize TensorRT engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.inference_fn = do_inference if trt.__version__[0] < '7' \
                                         else do_inference_v2
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def detect(self, priors, cfg, img):
        """Detect objects in the input image."""
        img_resized = _preprocess(img, self.input_shape)

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(img_resized)
        if self.cuda_ctx:
            self.cuda_ctx.push()
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        if self.cuda_ctx:
            self.cuda_ctx.pop()
        input_size = self.input_shape[0]
        output_size = int(2*(16+4+1)*(input_size/32)**2)
        output_shapes = [(1, output_size, 4), (1, output_size, 10), (1, output_size, 2)]
        loc, landms, conf = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]    
        facePositions, landmarks = _postprocess(
            loc, conf, landms, priors, cfg, img)

        return facePositions, landmarks
