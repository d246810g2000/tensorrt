import warnings
from nets.layers import Conv2D_BN_Leaky, Depthwise_Conv_Block

def MobileNet(inputs):
    # 320, 320, 3 -> 160, 160, 8
    x = Conv2D_BN_Leaky(inputs, 8, kernel_size=3, strides=2)
    # 160, 160, 8 -> 160, 160, 16
    x = Depthwise_Conv_Block(x, 16, strides=1)
    # 160, 160, 16 -> 80, 80, 32
    x = Depthwise_Conv_Block(x, 32, strides=2)
    x = Depthwise_Conv_Block(x, 32, strides=1)
    # 80, 80, 32 -> 40, 40, 64
    x = Depthwise_Conv_Block(x, 64, strides=2)
    x = Depthwise_Conv_Block(x, 64, strides=1)
    feat1 = x

    # 40, 40, 64 -> 20, 20, 128
    x = Depthwise_Conv_Block(x, 128, strides=2)
    for i in range(5):
        x = Depthwise_Conv_Block(x, 128, strides=1)
    feat2 = x

    # 20, 20, 128 -> 10, 10, 256
    x = Depthwise_Conv_Block(x, 256, strides=2)
    x = Depthwise_Conv_Block(x, 256, strides=1)    
    feat3 = x

    return feat1, feat2, feat3


