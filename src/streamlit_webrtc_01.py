import threading
from typing import Union
import json
import av
import cv2
import numpy as np
import streamlit as st
import torch
from torchvision import models
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class VideoTransformer(VideoTransformerBase):
    frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
    in_image: Union[np.ndarray, None]
    out_image: Union[np.ndarray, None]
    def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
            in_image = frame.to_ndarray(format="bgr24")
            
            out_image = in_image[:, ::-1, :]  # Simple flipping for example.

            with self.frame_lock:
                self.in_image = in_image
                self.out_image = out_image

            return out_image


        

ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)

if ctx.video_transformer:
    if st.button("Snapshot"):
        with ctx.video_transformer.frame_lock:
            in_image = ctx.video_transformer.in_image
            out_image = ctx.video_transformer.out_image
        
            # save image
            cv2.imwrite('D:\Research-work-2022\Streamlit\images\detect.jpg',out_image)