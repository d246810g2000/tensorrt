# Facerecognition on tensorrt
[InsightFace](https://github.com/deepinsight/insightface) 是一個人臉識別的方法，其方法是由 deepinsight (洞見實驗室) 所實現，是當時所有開源方法中的冠軍，而 deepinsight 為 WIDER FACE 標註了 landmark。我使用的模型如下，為了輕量其 backbone 皆為 mobilenet 
- 人臉偵測：[RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- 人臉識別：[MobileFaceNet (ArcFace)](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace)

![](https://i.imgur.com/WonulYz.png)

![](https://i.imgur.com/t03izAZ.png)
