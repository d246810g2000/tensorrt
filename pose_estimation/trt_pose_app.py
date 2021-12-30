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


import sys
import os
import cv2
import pose_capture
import video_app_utils
import argparse
import logging


class ColorConvert(video_app_utils.PipelineWorker):
    
    def __init__(self, qsize, source):
        super().__init__(qsize, source)
        
    def process(self, srcData):
        orgFrame = srcData
        frame = cv2.cvtColor(orgFrame, cv2.COLOR_BGR2RGB)
        return (True, (frame, orgFrame))

        
class Resize(video_app_utils.PipelineWorker):

    def __init__(self, qsize, source, inRes):
        super().__init__(qsize, source)
        self.inRes = inRes
        
    def process(self, srcData):
        frame, orgFrame = srcData
        frame = cv2.resize(frame, self.inRes, interpolation=cv2.INTER_NEAREST)
        return (True, (frame, orgFrame))
        

'''       
class ResizeGpu(video_app_utils.PipelineWorker):

    def __init__(self, qsize, source, inRes):
        super().__init__(qsize, source)
        self.inRes = inRes
        
    def process(self, srcData):
        frame, orgFrame = srcData
        srcGpu = cv2.cuda_GpuMat()
        dstGpu = cv2.cuda_GpuMat()
        srcGpu.upload(frame)
        dstGpu = cv2.cuda.resize( \
            srcGpu, self.inRes, interpolation=cv2.INTER_NEAREST)
        frame = dstGpu.download()
        return (True, (frame, orgFrame))
'''


class Preprocess(video_app_utils.PipelineWorker):
    
    def __init__(self, qsize, source, model):
        super().__init__(qsize, source)
        self.model = model
        
    def process(self, srcData):
        frame, orgFrame = srcData
        frame = self.model.preprocess(frame)
        return (True, (frame, orgFrame))

        
class Inference(video_app_utils.PipelineWorker):
    
    def __init__(self, qsize, source, model):
        super().__init__(qsize, source)
        self.model = model
        
    def process(self, srcData):
        frame, orgFrame = srcData
        cmap, paf = self.model.infer(frame)
        return (True, (cmap, paf, orgFrame))

        
class Postprocess(video_app_utils.PipelineWorker):
    
    def __init__(self, qsize, source, model):
        super().__init__(qsize, source)
        self.model = model
        self.cont = True
        
    def process(self, srcData):
        cmap, paf, orgFrame = srcData
        if not self.cont:
            return (True, None)
        self.cont = self.model.postprocess(cmap, paf, orgFrame)
        return (True, orgFrame)


class PoseEstimationProcess(video_app_utils.ContinuousVideoProcess):

    def __init__(self, args):
        '''
        [Capture]->[Color Convert]->[Resize]->[Pre-process]->
        ->[Infer]->[Post-process]->[Display]
        '''
        super().__init__(args)
        model = pose_capture.PoseCaptureModel( \
            args.model, args.task, args.csv, args.csvpath)
        inRes = model.getInputRes()
        colorConv = ColorConvert(args.qsize, self.capture)
        resize = Resize(args.qsize, colorConv, inRes)
        preprocess = Preprocess(args.qsize, resize, model)    
        inference = Inference(args.qsize, preprocess, model)  
        postprocess = Postprocess(args.qsize, inference, model)
  
        
def main():
    # Parse the command line parameters
    cvpParser = video_app_utils.ContinuousVideoProcess.argumentParser( \
        width=640, height=480)
    parser = argparse.ArgumentParser( \
        parents=[cvpParser], description='TRT Pose Demo')
    parser.add_argument('--model', \
        type=str, \
        default='resnet18_baseline_att_224x224_A_epoch_249.pth', \
        metavar='MODEL', \
        help='Model weight file')
    parser.add_argument('--task', \
        type=str, \
        default='human_pose.json', \
        metavar='TASK_DESC', \
        help='Task description file')
    parser.add_argument('--csv', \
        type=int, \
        default=0, \
        metavar='MAX_CSV_REC', \
        help='Maximum CSV records')
    parser.add_argument('--csvpath', \
        type=str, \
        default=os.path.join('.', 'csv'), \
        metavar='CSV_PATH', \
        help='Directory path to save CSV files')
    parser.add_argument('--verbose', \
        action='store_true', \
        help='If set, print debug message')
    args = parser.parse_args()
    # Set the logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    # Create continuous video process and start it
    try :
        vproc = PoseEstimationProcess(args)
        vproc.execute()
    except pose_capture.PoseCaptureError as err:
        print('Application error: %s' % (str(err)))
    except video_app_utils.VideoAppUtilsError as err:
        print('Video application framewrok error: %s' % (str(err)))
    

if __name__ == '__main__':
    main() 
    sys.exit(0)   
