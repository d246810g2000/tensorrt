# TensorRT
TensorRT 的核心是一個 c++ 的 library，透過 TensorRT 將 training framework 最佳化成一個 inference engine，這個 engine 能夠高效率的於 Nvidia GPU 進行 inference。

![](https://i.imgur.com/4dmIIlr.png)

如今 TensorRT 已經支援了很多深度學習的框架，但是有些框架需先轉換成 ONNX 的通用深度學習模型 (請參考 [onnx](https://github.com/d246810g2000/tensorrt/tree/main/onnx) 轉換模型教學)，才可以透過 TensorRT 進行最佳化，像是 Caffe2/CNTK/Chainer/PyTorch/MxNet 等等，而有些框架則已經把 TensorRT 集成到框架中了，像是 TensorFlow/MATLAB 等等。

![](https://i.imgur.com/zS3hjaI.png)


為了最佳化模型需先經過 TensorRT 的 network definition 再 build 成 engine，這個 build 的過程會需要一段時間，因此可以透過 serialize 將 engine 存成 trt 檔，下次若要使用可以透過 deserialize 將 trt 檔還原成 engine 以供預測。

![](https://i.imgur.com/6wt1qz1.png)
