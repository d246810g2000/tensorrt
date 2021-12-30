#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# MIT License
#
# Copyright (c) 2019, 2020 MACNICA Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE. 
#


import os
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import cv2
import torchvision.transforms as transforms
from draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import time
import argparse
import numpy as np
import PIL.Image
import csv
import datetime
import re
import logging


class PoseCaptureError(Exception):
    pass


class PoseCaptureDescError(PoseCaptureError):
    pass
    
    
class PoseCaptureModelError(PoseCaptureError):
    pass


class PoseCaptureCsvError(PoseCaptureError):
    pass
    

class PoseCaptureModel():
    
    def __init__(self, modelFile, taskDescFile, csv=0, csvPath='.'):
        
        # Load the task description
        try:
            with open(taskDescFile, 'r') as f:
                human_pose = json.load(f)
        except OSError:
            raise PoseCaptureDescError
        topology = trt_pose.coco.coco_category_to_topology(human_pose)    
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        
        # Load the base model
        fbase = os.path.basename(modelFile)
        func, self.inWidth, self.inHeight = \
            PoseCaptureModel.getModelFuncName(fbase)
        if func is None:
            logging.fatal('Invalid model name: %s' % (fbase))
            logging.fatal('Model name should be (.+_.+_att)_(\\d+)x(\\d+)_')
            raise PoseCaptureModelError('Invalid model name: %s' % (fbase))
        if not hasattr(trt_pose.models, func):
            logging.fatal('Could not find base model function: %s' % (func))
            raise PoseCaptureModelError( \
                'Could not find base model function: %s' % (func))
        func = 'trt_pose.models.' + func
        trtFile = os.path.splitext(fbase)[0] + '_trt.pth'
        logging.info('Loading base model from %s' % (func))
        model = eval(func)(num_parts, 2 * num_links).cuda().eval()
        
        if os.path.exists(trtFile) :
            logging.info('Loading model from TensorRT plan file ...')
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trtFile))
        else:
            logging.info('Optimizing model for TensorRT ...')
            model.load_state_dict(torch.load(modelFile))
            data = torch.zeros((1, 3, self.inHeight, self.inWidth)).cuda()
            model_trt = torch2trt.torch2trt( \
                model, [data], fp16_mode=True, max_workspace_size=1<<25)
            torch.save(model_trt.state_dict(), trtFile)
        
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')
        
        self.parse_objects = ParseObjects(topology)
        self.draw_objects = DrawObjects(topology)
        self.model_trt = model_trt
        self.num_parts = num_parts
        self.csv = csv
        self.count = 0

        if self.csv > 0:
            try:
                self._initCsv(human_pose['keypoints'], csvPath)
            except OSError:
                raise PoseCaptureCsvError

    def __del__(self):
        if hasattr(self, 'csv') and self.csv > 0:
            self._closeCsv()

    def _initCsv(self, keypoints, csvPath):
        if not os.path.exists(csvPath):
            os.mkdir(csvPath)
        fname = str(datetime.datetime.now()) + '.csv'
        fname = os.path.join(csvPath, fname)
        logging.info('CSV file: %s' % (fname))
        self.csvFile = open(fname, 'w')
        self.csvWriter = csv.writer(self.csvFile)
        labels = []
        labels.append('timestamp')
        labels.append('object_id')
        labels_x = [pt + '_x' for pt in keypoints]
        labels_y = [pt + '_y' for pt in keypoints]
        labels_pt = labels_x + labels_y
        labels_pt[::2] = labels_x
        labels_pt[1::2] = labels_y
        labels = labels + labels_pt
        self.csvWriter.writerow(labels)

    def _closeCsv(self):
        if hasattr(self, 'csvFile'):
            if self.csvFile is not None:
                self.csvFile.close()
                self.csvFile = None
        
    def preprocess(self, image):
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]
    
    def infer(self, image):
        cmap, paf = self.model_trt(image)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        return (cmap, paf)
    
    def postprocess(self, cmap, paf, image):
        counts, objects, peaks = self.parse_objects(cmap, paf)
        pt_lists = \
            [[0] * (self.num_parts * 2 + 2) for i in range(int(counts[0]))]
        dt = str(datetime.datetime.now())
        for i in range(int(counts[0])):
            pt_lists[i][0] = dt
            pt_lists[i][1] = i
        self.draw_objects(image, counts, objects, peaks, pt_lists)
        if self.csv > 0:
            self.csvWriter.writerows(pt_lists)
        self.count += 1
        if self.csv <= 0:
            return True
        elif self.count >= self.csv:
            logging.info('CSV recored was reached to the max value %d' \
                % (self.csv))
            return False
        else:
            return True
            
    def getInputRes(self):
        return (self.inWidth, self.inHeight)
        
    @staticmethod
    def getModelFuncName(model_file):
        result = re.search('(.+_.+_att)_(\d+)x(\d+)_', model_file)
        if result is None:
            return (None, None, None)
        else:
            return (result[1], int(result[2]), int(result[3]))
