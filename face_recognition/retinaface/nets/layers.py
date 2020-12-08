import warnings
from tensorflow.keras.layers import (
    DepthwiseConv2D, BatchNormalization, LeakyReLU, Conv2D, Input, Add, UpSampling2D,
    Concatenate, Activation, Reshape)

def Conv2D_BN(inputs, filters, kernel_size=3, strides=1):
    x = Conv2D(filters, 
               kernel_size,
               padding='same',
               strides=strides,
               use_bias=False)(inputs)
    x = BatchNormalization()(x)
    return x

def Conv2D_BN_Leaky(inputs, filters, kernel_size=3, strides=1, leaky=0.1):
    x = Conv2D(filters, 
               kernel_size,
               padding='same',
               strides=strides,
               use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leaky)(x)
    return x

def Depthwise_Conv_Block(inputs, pointwise_conv_filters, strides=1):  
    x = DepthwiseConv2D(kernel_size=3, 
                        padding="same",
                        strides=strides,
                        use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=pointwise_conv_filters,
               kernel_size=1, 
               padding="same",
               strides=1,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

def SSH(inputs, out_channel):
    conv3X3 = Conv2D_BN(inputs, out_channel//2, kernel_size=3, strides=1)

    conv5X5_1 = Conv2D_BN_Leaky(inputs, out_channel//4, kernel_size=3, strides=1)
    conv5X5 = Conv2D_BN(conv5X5_1, out_channel//4, kernel_size=3, strides=1)

    conv7X7_2 = Conv2D_BN_Leaky(conv5X5_1, out_channel//4, kernel_size=3, strides=1)
    conv7X7 = Conv2D_BN(conv7X7_2, out_channel//4, kernel_size=3, strides=1)

    outputs = Concatenate()([conv3X3, conv5X5, conv7X7])
    outputs = Activation('relu')(outputs)
    return outputs

def ClassHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors*2, kernel_size=1, strides=1)(inputs)
    outputs = Activation('softmax')(Reshape([-1, 2])(outputs))
    return outputs

def BboxHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors*4, kernel_size=1, strides=1)(inputs)
    outputs = Reshape([-1, 4])(outputs)
    return outputs

def LandmarkHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors*5*2, kernel_size=1, strides=1)(inputs)
    outputs = Reshape([-1, 10])(outputs)
    return outputs