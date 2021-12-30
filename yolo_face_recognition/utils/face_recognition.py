"""yolo_with_plugins.py

Implementation of TrtYOLO class with the yolo_layer plugins.
"""


from __future__ import print_function

import numpy as np
import cv2
import pickle
import tensorrt as trt
import pycuda.driver as cuda
from sklearn.preprocessing import normalize

def _aligned_face(img, bbox):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_size = (112, 112)
    det = bbox
    margin = 20
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
    bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = cv2.resize(ret, (112, 112))
    aligned = aligned.transpose((2, 0, 1)).astype(np.float32)
    return aligned

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class TRT_MobileFacenet(object):
    """TrtYOLO class encapsulates things needed to run TRT YOLO."""

    def _load_engine(self):
        TRTbin = 'mobilefacenet/mobilefacenet_fp16.trt'
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

    def face_encodings(self, img, facePositions):
        embeddings = np.zeros((len(facePositions), 128))        
        for idx, facePosition in enumerate(facePositions):
            aligned = _aligned_face(img, facePosition)
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