# tensorrt
先安裝一些相依套件，並下載 nanodet 的 repo
```
$ pip3 install Cython termcolor numpy tensorboard pycocotools matplotlib pyaml opencv-python tqdm
$ git clone https://github.com/RangiLyu/nanodet.git
$ cd nanodet
$ python setup.py develop
```
下載 nanodet 的 coco pretrained weight
```
$ pip3 install gdown
$ gdown --id '1EhMqGozKfqEfw8y9ftbi1jhYu86XoW62'
```
將 pth 轉成 onnx，運行 tools 裡的 export.py 可以得到 output.onnx
```
!python3 tools/export.py
```
原本的 pth 網路架構輸出層是輸出 3 個 scale：10x10, 20x20, 40x40
- class: [1, 80, 10, 10], [1, 80, 20, 20], [1, 80, 40, 40]
- bbox: [1, 32, 10, 10], [1, 32, 20, 20], [1, 32, 40, 40]

轉完的 onnx 會經過 sigmoid 和 reshape 變成 (在 pytorch 中在後處理時會經過這些步驟)
- class: [1, 100, 80], [1, 400, 80], [1, 1600, 80]
- bbox: [1, 100, 32], [1, 400, 32], [1, 1600, 32]

我們可以將轉好的 output.onnx 放到 netron 查看網路架構


![](https://i.imgur.com/n8nVmmz.png)

為了方便起見我們直接更改 outputs 的位置，我們使用 onnx_edit.py 來更改，--outpus 第一個位置放 output name，[]裡放 output shape，例如 831[1,80,10,10]，點選 node 可以查看資訊。
```
$ wget https://raw.githubusercontent.com/d246810g2000/tensorrt/main/onnx_edit.py
$ python3 onnx_edit.py output.onnx nanodet.onnx\
--outputs '787[1,80,40,40], 788[1,32,40,40], 809[1,80,20,20], 810[1,32,20,20], 831[1,80,10,10], 832[1,32,10,10]'
```
![](https://i.imgur.com/G7MwTZo.png)

更改後的網路架構如下

![](https://i.imgur.com/v7lJV0t.png)

將更改完的 nanodet.onnx 轉成 tensorrt 的 engine，預設是 FP16，batch_size 為 1

```
$ wget https://raw.githubusercontent.com/d246810g2000/tensorrt/main/build_engine.py
$ python3 build_engine.py -o nanodet.onnx -t nanodet.trt
```

接著我們就可以來 inference 了，試試用 tensorrt 優化後的 trt_demo.py，為了方便起見我直接將 nanodet.trt 的輸出轉換成 torch.tensor 再進行後處理，結果如下，FPS 在 5 左右。

```
# trt_demo.py 需要 import common
$ wget https://raw.githubusercontent.com/d246810g2000/tensorrt/main/nanodet/common.py
$ wget https://raw.githubusercontent.com/d246810g2000/tensorrt/main/nanodet/trt_demo.py
$ python3 demo/trt_demo.py csicam --config config/nanodet-m.yml --model nanodet.trt
```

![](https://i.imgur.com/MC90zhc.png)
