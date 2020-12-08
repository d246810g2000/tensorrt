"""face_recognition.py

Implementation of trt_face_recognition
"""

import cv2
import pickle
import numpy as np
import tensorrt as trt
from utils.common import *
import pycuda.driver as cuda
from skimage import transform as trans

def normalize(arr: np.array):
    x_norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr/x_norm
    return arr

def _aligned_face(img, landmark):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32)
    dst = np.array(landmark, dtype=np.float32).reshape(5, 2)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue = 0)
    aligned = aligned.transpose((2, 0, 1))
    return aligned

class TRT_MobileFacenet(object):
    """TRT_MobileFacenet class encapsulates things needed to run TRT recognition."""

    def _load_engine(self):
        TRTbin = 'mobilefacenet/mobilefacenet.trt'
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, db_file, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.db_file = db_file
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

    def face_encodings(self, img, landmarks):
        embeddings = np.zeros((len(landmarks), 128))        
        for idx, landmark in enumerate(landmarks):
            aligned = _aligned_face(img, landmark)
            # Set host input to the image. The do_inference() function
            # will copy the input to the GPU before executing.
            self.inputs[0].host = np.ascontiguousarray(aligned)
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
            embedding = trt_outputs[0].reshape(1, -1)
            embeddings[idx] = normalize(embedding).flatten()
        return embeddings

    def compare_faces(self, embedding, threshold):
        # read db
        with open(self.db_file, 'rb') as file:
            db = pickle.load(file)
        db_embeddings = db['embeddings']
        db_names = db['names']

        distances = np.zeros((len(db_embeddings)))
        for i, db_embedding in enumerate(db_embeddings):
            distance = round(np.linalg.norm(db_embedding-embedding), 2)
            distances[i] = distance
        idx_min = np.argmin(distances)
        distance, name = distances[idx_min], db_names[idx_min]
        if distance < threshold:
            return name, distance
        else:
            return False, distance

