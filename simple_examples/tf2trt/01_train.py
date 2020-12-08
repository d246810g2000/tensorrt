import tensorflow as tf 
from tensorflow.keras.applications.mobilenet import MobileNet
model = MobileNet(weights='imagenet',
                  include_top=True)
# save to saved_model
tf.saved_model.save(model, 'mobilenet_saved_model')