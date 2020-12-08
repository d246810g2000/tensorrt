import numpy as np
from math import ceil
from itertools import product as product

class Anchors(object):
    def __init__(self, cfg, phase='train'):
        super(Anchors, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = cfg['image_size']
        self.feature_maps = [[ceil(self.image_size/step), ceil(self.image_size/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # 每個網格有兩個 anchors，都是正方形
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size
                    s_ky = min_size / self.image_size
                    dense_cx = [x * self.steps[k] / self.image_size for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        # x, y, w, h
        anchors = np.reshape(anchors,[-1,4])
        
        # 將 x, y, w, h 轉換成 x1, y1, x2, y2
        output = np.zeros_like(anchors[:,:4])
        output[:,0] = anchors[:,0] - anchors[:,2]/2
        output[:,1] = anchors[:,1] - anchors[:,3]/2
        output[:,2] = anchors[:,0] + anchors[:,2]/2
        output[:,3] = anchors[:,1] + anchors[:,3]/2

        if self.clip:
            output = np.clip(output, 0, 1)
        return output


class BBoxUtility(object):
    def __init__(self, priors=None, overlap_threshold=0.35, nms_thresh=0.45):
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh

    def iou(self, box):
        """計算出每個 ground truth 和 anchors 的 iou
           
           # Arguments:
           box: x1, y1, x2, y2            
        """
        # 計算 ground truth 和 anchors 的交集面積
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        
        # ground truth 的面積
        area_gt = (box[2] - box[0]) * (box[3] - box[1])
        
        # anchors 的面積
        area_anchors = (self.priors[:, 2] - self.priors[:, 0])*(self.priors[:, 3] - self.priors[:, 1])
        
        # 計算 iou
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        union = area_anchors + area_gt - inter
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        """將 gt 與對應的 anchor 編碼成訓練格式
           
           # Arguments:
           box: x1, y1, x2, y2            
        """
        iou = self.iou(box[:4])

        encoded_box = np.zeros((self.num_priors, 4 + return_iou + 10))

        # 根據 gt，找到跟他重合程度較高的 anchors，
        # 若 iou < overlap_threshold，則匹配給他重合程度最高的 anchor
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, 4][assign_mask] = iou[assign_mask]
        
        # 找到對應的 anchor
        assigned_priors = self.priors[assign_mask]
        # 逆向編碼，將 gt 轉換為 efficientdet 預測結果的格式

        # 先計算 gt 的中心與長寬
        box_center = 0.5 * (box[:2] + box[2:4])
        box_wh = box[2:4] - box[:2]
        # 再計算重合度較高的 anchors 的中心與長寬
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        # 逆向求取 efficientdet 應該有的預測結果
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= 0.1
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= 0.2

        ldm_encoded = np.zeros_like(encoded_box[:, 5:][assign_mask])
        ldm_encoded = np.reshape(ldm_encoded, [-1, 5, 2])
        ldm_encoded[:, :, 0] = box[[4, 6, 8, 10, 12]] - np.repeat(assigned_priors_center[:, 0:1], 5, axis=-1)
        ldm_encoded[:, :, 1] = box[[5, 7, 9, 11, 13]] - np.repeat(assigned_priors_center[:, 1:2], 5, axis=-1)
        ldm_encoded[:, :, 0] /= np.repeat(assigned_priors_wh[:,0:1], 5, axis=-1)
        ldm_encoded[:, :, 1] /= np.repeat(assigned_priors_wh[:,1:2], 5, axis=-1)
        ldm_encoded[:, :, 0] /= 0.1
        ldm_encoded[:, :, 1] /= 0.1

        encoded_box[:, 5:][assign_mask] = np.reshape(ldm_encoded, [-1,10])
        # print(encoded_box[assign_mask])
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_priors, 4 + 1 + 2 + 1 + 10 + 1))
        assignment[:, 5] = 1
        if len(boxes) == 0:
            return assignment
            
        # (n, num_priors, 5)
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes)
        # 每一個 gt 編碼後的值和 iou
        # (n, num_priors)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 15)

        # 取重合程度最大的 anchor，並且獲取其 index
        # (num_priors)
        best_iou = encoded_boxes[:, :, 4].max(axis=0)
        # (num_priors)
        best_iou_idx = encoded_boxes[:, :, 4].argmax(axis=0)
        # (num_priors)
        best_iou_mask = best_iou > 0
        # 某個 anchor 他屬於哪個 gt
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的 anchor 與應有的預測結果
        # 哪些 anchors 存在 gt
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        assignment[:, 4][best_iou_mask] = 1

        assignment[:, 5][best_iou_mask] = 0
        assignment[:, 6][best_iou_mask] = 1
        assignment[:, 7][best_iou_mask] = 1

        assignment[:, 8:-1][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), 5:]
        assignment[:, -1][best_iou_mask] = boxes[best_iou_idx, -1]
        # 通過 assign_boxes 我們就獲得了，輸入進來的這張照片，應該有的預測結果是什麼樣子的

        return assignment