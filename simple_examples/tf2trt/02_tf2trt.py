import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
saved_file = 'mobilenet_saved_model'
TRT_file = 'mobilenet_saved_model_TFTRT_FP16'
print('Converting to TF-TRT FP16...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(
    max_workspace_size_bytes=(1<<28),
    precision_mode="FP16",
    maximum_cached_engines=1)
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=saved_file,
    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir=TRT_file)
print('Done Converting to TF-TRT FP16')