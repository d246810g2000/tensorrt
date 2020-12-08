from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D, Add, Concatenate
from nets.layers import Conv2D_BN_Leaky, SSH
from nets.layers import ClassHead, BboxHead, LandmarkHead
from nets.mobilenet025 import MobileNet

def Retinaface(cfg):
    inputs = Input(shape=(cfg['image_size'], cfg['image_size'], 3))
    C3, C4, C5 = MobileNet(inputs)
    P3 = Conv2D_BN_Leaky(C3, cfg['out_channel'], kernel_size=1, strides=1)
    P4 = Conv2D_BN_Leaky(C4, cfg['out_channel'], kernel_size=1, strides=1)
    P5 = Conv2D_BN_Leaky(C5, cfg['out_channel'], kernel_size=1, strides=1)

    # Feature Pynamid Network
    P5_upsampled = UpSampling2D(size=(2, 2))(P5)
    P4 = Add()([P5_upsampled, P4])
    P4 = Conv2D_BN_Leaky(P4, cfg['out_channel'], kernel_size=3, strides=1)
    P4_upsampled = UpSampling2D(size=(2, 2))(P4)
    P3 = Add()([P4_upsampled, P3])
    P3 = Conv2D_BN_Leaky(P3, cfg['out_channel'], kernel_size=3, strides=1)

    # SSH
    SSH1 = SSH(P3, cfg['out_channel'])
    SSH2 = SSH(P4, cfg['out_channel'])
    SSH3 = SSH(P5, cfg['out_channel'])
    SSH_all = [SSH1, SSH2, SSH3]

    # outputs
    bbox_regressions = Concatenate(axis=1, name="bbox_reg")([BboxHead(feature) for feature in SSH_all])
    classifications = Concatenate(axis=1, name="cls")([ClassHead(feature) for feature in SSH_all])
    landm_regressions = Concatenate(axis=1, name="landm_reg")([LandmarkHead(feature) for feature in SSH_all])
    outputs = [bbox_regressions, classifications, landm_regressions]
    model = Model(inputs=inputs, outputs=outputs)
    return model