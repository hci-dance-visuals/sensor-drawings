import io
from tornado import ioloop
from PIL import Image
import numpy as np
import cv2

class PyImageStreamer:
    def __init__(self, quality, stopdelay, port, scalar):
        print("Initializing PyImageStreamer...")
        # self._cam = cv2.VideoCapture(0)
        self.is_started = False
        self.stop_requested = False
        self.quality = quality
        self.stopdelay = stopdelay
        self.port = port
        self.scalar = scalar

    def request_start(self):
        if self.stop_requested:
            print("PyImageStreamer continues to be in use")
            self.stop_requested = False
        if not self.is_started:
            self._start()

    def request_stop(self):
        if self.is_started and not self.stop_requested:
            self.stop_requested = True
            print("Stopping PyImageStreamer in " + str(self.stopdelay) + " seconds...")
            ioloop.IOLoop.current().call_later(self.stopdelay, self._stop)

    def _start(self):
        print("Starting PyImageStreamer...")
        self.is_started = True

    def _stop(self):
        if self.stop_requested:
            print("Stopping PyImageStreamer now...")
            print("PyImageStreamer stopped")
            self.is_started = False
            self.stop_requested = False

    def get_jpeg_image_bytes(self, frame):
        cv2_im = frame
        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_RGB2GRAY)
        cv2_im = cv2.resize(cv2_im, (0,0), fx=self.scalar, fy=self.scalar)
        pil_im = Image.fromarray(cv2_im)
        with io.BytesIO() as bytesIO:
            with pil_im as img:
#                img.save(bytesIO, 'JPEG')
                img.save(bytesIO, "PNG", quality=self.quality, optimize=True)
            return bytesIO.getvalue()
