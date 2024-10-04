# Import kits
import cv2
import os
import numpy
import pandas as pd
from moviepy.editor import *
import numpy as np
from CNN_for_shapes_data import resort_file
from time import sleep

# 剪辑50-60秒的音乐 00:00:50 - 00:00:60
video = CompositeVideoClip([VideoFileClip("Z:/Project-NC-2023-A-02/1 - data/Rats/R701/20231129/Behavioral/Video4/20231129R701_labeled.mp4").subclip(((2413/1800)+1)*60, ((2413/1800)+3)*60)])
# 写入剪辑完成的音乐
video.write_videofile("Z:/Project-NC-2023-A-02/1 - data/Rats/R701/20231129/Behavioral/Video4/20231129R701_cut.mp4")
