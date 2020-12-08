# Simple Examples
1. Keras (tf.keras) -> ONNX -> TensorRT
2. TensorFlow -> TensorRT
![](https://i.imgur.com/3MjX4k7.png)

## TF -> ONNX -> TRT
- 優點：消耗更少的內存，推論速度通常比 TF-TRT 快
- 缺點：若遇到某些圖層不支持 TRT，我們必須為這些圖層使用插件 (plugin) 或自定義代碼來實現

## TF -> TRT
- 優點：API 易於使用，若遇到某些圖層不支持 TRT 會自動由 TF 執行，無需擔心插件 (plugin)。
- 缺點：需要將整個 TF 庫存儲在 HDD 中（這對部署環境不利）；需要在運行時將 TF 加載到內存中，通常比純 TRT 引擎運行慢。
