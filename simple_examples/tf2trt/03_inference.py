import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

def preprocess(img_path, img_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    img = np.float32(img)/255.0
    img = cv2.resize(img, (img_size, img_size))
    return img

# 讀進照片並進行前處理
img_path = 'tabby_tiger_cat.jpg'
img_size = 224
img = preprocess(img_path, img_size)

input_saved_model = 'mobilenet_saved_model_TFTRT_FP16'
saved_model_loaded = tf.saved_model.load(
	input_saved_model,
    tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
print(signature_keys)

infer = saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)
keys = list(infer.structured_outputs.keys())

preds = infer(img)[keys[0]].numpy()
print('Predicted: {}'.format(decode_predictions(preds, top=1)[0]))

