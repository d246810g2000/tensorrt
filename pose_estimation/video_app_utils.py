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

'''A collection of utility classes for video applications.
'''

import sys
import queue
import threading
import cv2
import time
import numpy as np
import argparse
import datetime
import logging


class VideoAppUtilsError(Exception):
    pass


class VideoAppUtilsEosError(VideoAppUtilsError):
    pass
        

class VideoAppUtilsDeviceError(VideoAppUtilsError):
    pass


class PipelineWorker():
    '''A worker thread for a stage in a software pipeline.
    This class is an abstruct class. Sub classes inherited from this class
    should implement a process stage of a software pipeline.
    
    +------------------+ +------------------+ +------------------+
    | PipelineWorker#1 | | PipelineWorker#2 | | PipelineWorker#3 |<-get()
    | getData()->(Q)-----> process()->(Q)-----> process()->(Q)----->
    +------------------+ +------------------+ +------------------+
    
    Attributes:
        queue: Queue to store outputs processed by this instance.
        source: Data source (assumped to be other PipelineWorker instance) 
        destination: Data destination
                     (assumped to be other PipelineWorker instance)  
        sem: Semaphore to lock this instance.
        flag: If ture, the processing loop is running.
        numDrops: Total number of dropped outputs.
        thread: Worker thread runs the _run instance method.
    '''
    
    def __init__(self, qsize, source=None, drop=True):
        '''
        Args:
            qsize(int): Output queue capacity
            source(PipelineWorker): Data source. If ommited, derived class
                should implement the getData method.
        '''
        self.queue = queue.Queue(qsize)
        self.source = source
        if self.source is not None:
            self.source.destination = self
        self.drop = drop
        self.destination = None
        self.sem = threading.Semaphore(1)
        self.flag = False
        self.numDrops = 0
        self._error = False
    
    def __del__(self):
        pass

    def __repr__(self):
        return '%02d %06d' % (self.qsize(), self.numDrops)
        
    def process(self, srcData):
        '''Data processing(producing) method called in thread loop.
        Derived classes should implement this method.
        
        Args:
            srcData: Source data 
        '''
        return (False, None)
        
    def getData(self):
        '''Returns a output to data consumer.
        '''
        if self.source is None:
            return None
        else:
            return self.source.get()
        
    def __run(self):
        logging.info('%s thread started' % (self.__class__.__name__))
        with self.sem:
            self.flag = True
        while True:
            with self.sem:
                if self.flag == False:
                    break
            dat = None
            try:
                src = self.getData()
            except VideoAppUtilsEosError:
                self._error = True
                logging.info('End of Stream detected')
            else:
                try:
                    ret, dat = self.process(src)
                    if ret == False:
                        self._error = True
                        dat = None
                        logging.info('Processing error')
                except Exception as e:
                    self._error = True
                    dat = None
                    logging.critical(e)
            if self.drop and self.queue.full():
                self.queue.get(block=True)
                self.numDrops += 1
            self.queue.put(dat)
        logging.info('%s thread terminated' % (self.__class__.__name__))
               
    def clear(self):
        try:
            while True:
                self.queue.get(block=False)
        except queue.Empty:
            return
    
    def start(self): 
        '''Starts the worker thread.
        '''     
        self.thread = threading.Thread(target=self.__run)
        self.thread.start()
        
    def get(self):
        '''Gets a output.
        '''
        if self._error:
            logging.info('VideoAppUtilsEosError')
            raise VideoAppUtilsEosError
            return None
        return self.queue.get(block=True)
        
    def stop(self):
        '''Stops the worker thread.
        '''
        with self.sem:
            self.flag = False
        self.clear()
        self.thread.join()
        
    def qsize(self):
        '''Returns the number of the current queued outputs
        '''
        sz = 0
        with self.sem:
            sz = self.queue.qsize()
        return sz
        

class ContinuousVideoCapture(PipelineWorker):
    '''Video capture workeer thread
    '''

    GST_STR_CSI = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, \
    format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv flip-method=0 \
    ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
    
    def __init__(self, cameraId, width, height, \
        fps=None, qsize=30, fourcc=None):
        '''
            Args:
                cameraId(int): Camera device ID, if negative number specified,
                    the CSI camera will be selected.
                width(int): Capture width
                height(int): Capture height
                fps(int): Frame rate
                qsize(int): Capture queue capacity
                fourcc(str): Capture format FOURCC string
        '''
        
        super().__init__(qsize)
        
        if cameraId < 0:
            # CSI camera
            if fps is None:
                fps = 30
            gstCmd = ContinuousVideoCapture.GST_STR_CSI \
                % (width, height)
            self.capture = cv2.VideoCapture(gstCmd, cv2.CAP_GSTREAMER)
            if self.capture.isOpened() is False:
                raise VideoAppUtilsDeviceError( \
                    'CSI camera could not be opened.')
        else:
            # USB camera
            # Open the camera device
            self.capture = cv2.VideoCapture(cameraId)
            if self.capture.isOpened() is False:
                raise VideoAppUtilsDeviceError( \
                    'Camera %d could not be opened.' % (cameraId))
                
            # Set the capture parameters
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fourcc is not None:
                self.capture.set(cv2.CAP_PROP_FOURCC, \
                    cv2.VideoWriter_fourcc(*fourcc))
            if fps is not None:
                self.capture.set(cv2.CAP_PROP_FPS, fps)
        
        # Get the actual frame size
        # Not work for OpenCV 4.1
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def __del__(self):
        super().__del__()
        self.capture.release()
        
    def getData(self):
        ret, frame = self.capture.read()
        if ret == False:
            raise VideoAppUtilsEosError
        return frame
        
    def process(self, srcData):
        return (True, srcData)


class VideoDecoder(PipelineWorker):
    
    GST_STR_DEC_H264 = 'filesrc location=%s \
    ! qtdemux name=demux demux.video_0 \
    ! queue \
    ! h264parse \
    ! omxh264dec \
    ! nvvidconv \
    ! video/x-raw, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
    
    GST_STR_DEC_H265 = 'filesrc location=%s \
    ! qtdemux name=demux demux.video_0 \
    ! queue \
    ! h265parse \
    ! omxh265dec \
    ! nvvidconv \
    ! video/x-raw, format=(string)BGRx \
    ! videoconvert \
    ! appsink'

    def __init__(self, file, qsize=30, repeat=False, h265=False):
        '''
            Args:
                file(str): 
                qsize(int): Capture queue capacity
        '''
        
        super().__init__(qsize)
        self.repeat = repeat
        if h265:
            self.gstCmd = VideoDecoder.GST_STR_DEC_H265 % (file)
        else:
            self.gstCmd = VideoDecoder.GST_STR_DEC_H264 % (file)
        self.capture = cv2.VideoCapture(self.gstCmd, cv2.CAP_GSTREAMER)
        if self.capture.isOpened() == False:
            raise VideoAppUtilsEosError('%s could not be opened.' % (file))
        
        # Get the frame size
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames = 0
    
    def __del__(self):
        super().__del__()
        self.capture.release()
        
    def getData(self):
        ret, frame = self.capture.read()
        if ret == False:
            if self.repeat:
                # Reopen the video file
                self.capture.release()
                self.capture = cv2.VideoCapture(self.gstCmd, cv2.CAP_GSTREAMER)
                if self.capture.isOpened() == False:
                    raise VideoAppUtilsEosError( \
                        '%s could not be re-opened.' % (file))
                    frame = None
                ret, frame = self.capture.read()
                if ret == False:
                    raise VideoAppUtilsEosError
            else:
                logging.info('End of stream at frame %d' % (self.frames))
                raise VideoAppUtilsEosError
        self.frames += 1
        return frame
        
    def process(self, srcData):
        return (True, srcData)


class IntervalCounter():
    '''A counter to measure the interval between the measure method calls.
    
    Attributes:
        numSamples: Number of samples to calculate the average.
        samples: Buffer to store the last N intervals.
        lastTime: Last time stamp
        count: Total counts
    '''

    def __init__(self, numSamples):
        '''
        Args:
            numSamples(int): Number of samples to calculate the average.
        '''
        self.numSamples = numSamples
        self.samples = np.zeros(self.numSamples)
        self.lastTime = time.time()
        self.count = 0
        
    def __del__(self):
        pass
        
    def measure(self):
        '''Measure the interval from the last call.
        
        Returns:
            The interval time count in second.
            If the number timestamps captured in less than numSamples,
            None will be returned.
        '''
        curTime = time.time()
        elapsedTime = curTime - self.lastTime
        self.lastTime = curTime
        self.samples = np.append(self.samples, elapsedTime)
        self.samples = np.delete(self.samples, 0)
        self.count += 1
        if self.count > self.numSamples:
            return np.average(self.samples)
        else:
            return None


class ContinuousVideoProcess():
    '''Captured video processing applicaion framework
    
    Attributes:
        capture(ContinuousVideoCapture): Video capture process
        fpsCounter(IntervalCounter): FPS counter
        qinfo(bool): If set, print processing queue status
        title(str): Window title
        pipeline(list): List of the pipeline worker objects
    '''
    
    def __init__(self, args):
        '''
        Args:
            args(argparse.Namespace): video capture command-line arguments
        '''
        if args.src_file is not None:
            self.capture = VideoDecoder( \
                args.src_file, args.qsize, args.repeat, args.h265)
        else:
            fourcc = None
            if args.mjpg:
                fourcc = 'MJPG'
            if args.fps < 0:
                fps = None
            else:
                fps = args.fps
            self.capture = ContinuousVideoCapture( \
                args.camera, args.width, args.height, fps, args.qsize, fourcc)
        self.fpsCounter = IntervalCounter(10)
        self.qinfo = args.qinfo
        self.title = args.title
        self.nodrop = args.nodrop
        self.pipeline = None
        
    def __del__(self):
        cv2.destroyAllWindows()
        self.stopPipeline()

    def scanPipeline(self):
        pipeline = []
        self.__class__.getSources(self.capture, pipeline)
        self.pipeline = pipeline[::-1]

    def startPipeline(self):
        self.scanPipeline()
        for worker in self.pipeline:
            if self.nodrop:
                worker.drop = False
            worker.start()

    def stopPipeline(self):
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            for worker in self.pipeline:
                worker.stop()

    def execute(self):
        ''' execute video processing loop
        '''
        self.startPipeline()
        while True:
            frame = self.getOutput()
            if frame is None:
                break
            if self.qinfo:
                print(self.pipeline)
            interval = self.fpsCounter.measure()
            if interval is not None:
                fps = 1.0 / interval
                dt = datetime.datetime.now().strftime('%F %T')
                fpsInfo = '{0}{1:.2f} {2}'.format('FPS:', fps, dt)
                cv2.putText(frame, fpsInfo, (8, 32), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(self.title, frame)  
            # Check if ESC key is pressed to terminate this application
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break
            # Check if the window was closed
            if cv2.getWindowProperty(self.title, cv2.WND_PROP_AUTOSIZE) < 0:
                break
        cv2.destroyAllWindows()
        self.stopPipeline()

    def getOutput(self):
        ''' Get the output image to be displayed.

        Returns:
            Output image
        '''
        try:
            frame = (self.pipeline[0]).get()
        except VideoAppUtilsEosError:
            return None
        else:
            return frame

    @staticmethod
    def getSources(worker, srcList):
        if worker is None:
            return
        else:
            srcList.append(worker)
        ContinuousVideoProcess.getSources(worker.destination, srcList)

    @staticmethod
    def argumentParser(**kwargs):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--camera', '-c', \
            type=int, \
            default=0, \
            metavar='CAMERA_NUM', \
            help='Camera number, use any negative integer for MIPI-CSI')
        parser.add_argument('--width', \
            type=int, \
            default=640, \
            metavar='WIDTH', \
            help='Capture width')
        parser.add_argument('--height', \
            type=int, \
            default=480, \
            metavar='HEIGHT', \
            help='Capture height')
        parser.add_argument('--fps', \
            type=int, \
            default=-1, \
            metavar='FPS', \
            help='Capture frame rate')
        parser.add_argument('--qsize', \
            type=int, \
            default=1, \
            metavar='QSIZE', \
            help='Capture queue size')
        parser.add_argument('--qinfo', \
            action='store_true', \
            help='If set, print queue status information')
        parser.add_argument('--mjpg', \
            action='store_true', \
            help='If set, capture video in motion jpeg format')
        parser.add_argument('--title', \
            type=str, \
            default=parser.prog, \
            metavar='TITLE', \
            help='Window title')
        parser.add_argument('--nodrop', \
            action='store_true', \
            help='If set, disable frame drop feature')
        parser.add_argument('--repeat', \
            action='store_true', \
            help='If set, repeat video decoding')
        parser.add_argument('--h265', \
            action='store_true', \
            help='If set, the specified video file will be assumed as H.265. \
                Otherwise, assumed as H.264')
        parser.add_argument('src_file', \
            type=str, \
            metavar='SRC_FILE', \
            nargs='?', \
            help='Source video file')
        parser.set_defaults(**kwargs)
        return parser


def player():
    # Parse the command line parameters
    cvpParser = ContinuousVideoProcess.argumentParser(title='Player')
    parser = argparse.ArgumentParser( \
        parents=[cvpParser], description='Simple Video/Camera Player')
    args = parser.parse_args()
    # Create continuous video process and start it
    vproc = ContinuousVideoProcess(args)
    vproc.execute()

