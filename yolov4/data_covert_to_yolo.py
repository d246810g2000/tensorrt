import os
import cv2
import json
import codecs
import requests
import numpy as np
from PIL import Image
from io import BytesIO

def get_data_from_json(address):
    jsonData = []
    with codecs.open(address, 'rU', 'utf-8') as js:
        for line in js:
            jsonData.append(json.loads(line))
    return jsonData

def convert_to_yolo(jsonData, yolo_path, train_val_split):
    count = 1
    totalfaces = 0
    split_data = len(jsonData)*(1-train_val_split)
    now_path = os.getcwd()
    for data in jsonData:
        response = requests.get(data['content'])
        img = np.asarray(Image.open(BytesIO(response.content)))
        if img.ndim==3:
            cv2.imwrite(f'{yolo_path}{count}.jpg', img[:,:,::-1])
            print("Processing "+ str(count)+'/'+str(len(jsonData)) + " images...")
            for annot in data["annotation"]:
                points = annot['points']
                if 'Face' in annot['label']:
                    x = (points[0]['x'] + points[1]['x'])/2
                    y = (points[0]['y'] + points[1]['y'])/2
                    w = points[1]['x'] - points[0]['x']
                    h = points[1]['y'] - points[0]['y']
                    text = "{} {} {} {} {}".format(0, x, y, w, h)
                    totalfaces += 1
                    with open(yolo_path + str(count) + '.txt', 'a') as f:
                        f.write(f'0 {x} {y} {w} {h}\n')
            if count < split_data:
                with open('data/train.txt', 'a') as f:
                    path = os.path.join(now_path, yolo_path)
                    line_txt = [path + str(count) + '.jpg', '\n']
                    f.writelines(line_txt)
            else:
                with open('data/val.txt', 'a') as f:
                    path = os.path.join(now_path, yolo_path)
                    line_txt = [path + str(count) + '.jpg', '\n']
                    f.writelines(line_txt)            
            count += 1    
        else:
            print(str(count)+'.jpg is not rgb img.')
            count += 1

    print("Total Faces Detected {}".format(totalfaces))
    
def main():
    yolo_path = 'face/'
    if not os.path.exists(yolo_path):
        os.mkdir(yolo_path)
    train_val_split = 0.1
    address = 'face_detection.json'
    jsonData = get_data_from_json(address)
    convert_to_yolo(jsonData, yolo_path, train_val_split)
    
if __name__ == '__main__':
    main()