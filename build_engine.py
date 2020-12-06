import os
import argparse
import tensorrt as trt

def parse_args():
    parser = argparse.ArgumentParser(
        description='build the TensorRT engine')
    parser.add_argument('-o', '--onnx_file', default=None, type=str,
                        help='path to onnx file')
    parser.add_argument('-t', '--trt_file', default=None, type=str,
                        help='output TensorRT engine')
    parser.add_argument('-m', '--model_data_type', default=16, type=int,
                        help='32 => float32, 16 => float16')
    parser.add_argument('-b', "--batch_size", default=1, type=int,
                        help='maximum batch size')
    return parser

def get_engine(args):
    def build_engine():
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = args.batch_size
            builder.fp16_mode = args.model_data_type==16
            # Parse model file
            if not os.path.exists(args.onnx_file):
                print('ONNX file {} not found.'.format(args.onnx_file))
                exit(0)
            print('Loading ONNX file from path {}...'.format(args.onnx_file))
            with open(args.onnx_file, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            print('Completed parsing of ONNX file')
            print('Building TensorRT engine from {}; this may take a while...'.format(args.onnx_file))
            print('    FP16 mode: {}'.format(args.model_data_type==16))
            print('    Max batch size: {}'.format(args.batch_size))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(args.trt_file, "wb") as f:
                f.write(engine.serialize())

    if os.path.exists(args.trt_file):
        print("tensorrt engine is already exists on {}".format(args.trt_file))
    else:
        build_engine()

def main(args):
    get_engine(args)

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    main(args)
