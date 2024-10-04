import cv2
import numpy as np
import pandas as pd

WorkingFolder = r'Z:\Project-NC-2023-A-02\1 - data\Rats\R905\20240117\Behavioral\Video'  # setting work-folder
# define names
filename = "20240117R905"
video = f"{filename}.mp4"
# saved filename
result_video = f"{filename}_labeled.mp4"
Path = f'{WorkingFolder}/{video}'  # define cutting space video
Path_result = f'{WorkingFolder}/{result_video}'  # define cutting space video

# read video
cap = cv2.VideoCapture(Path)
fps_video = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# get video parameter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoWriter = cv2.VideoWriter(Path_result, fourcc, fps_video, (frame_width, frame_height))

# read labels
labels = "BehavioralIndex"
labels = f"{labels}.mp4"
Path = f'{WorkingFolder}/{labels}'  # define cutting space video
BehavioralIndex = pd.read_csv(f'{WorkingFolder}/BehavioralIndex.csv')   # read csv
BehavioralIndex.rename(columns={'Unnamed: 0': 'frame'}, inplace=True)

# add labels
frame_id = -61
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_id += 1
        if frame_id >= 0 & frame_id < BehavioralIndex.shape[1]:
            # predicted description
            Behavior = BehavioralIndex.loc[frame_id, "Description"]
            cv2.putText(frame, f"Behavior(video prediction): {Behavior}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out0 = BehavioralIndex.loc[frame_id, "out0"]
            cv2.putText(frame, f"out0(Resting): {float('%.2f' % out0)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out1 = BehavioralIndex.loc[frame_id, "out1"]
            cv2.putText(frame, f"out1(Drinking): {float('%.2f' % out1)}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out2 = BehavioralIndex.loc[frame_id, "out2"]
            cv2.putText(frame, f"out2(Stepping): {float('%.2f' % out2)}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out3 = BehavioralIndex.loc[frame_id, "out3"]
            cv2.putText(frame, f"out3(Turning): {float('%.2f' % out3)}", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out4 = BehavioralIndex.loc[frame_id, "out4"]
            cv2.putText(frame, f"out4(Standing): {float('%.2f' % out4)}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out5 = BehavioralIndex.loc[frame_id, "out5"]
            cv2.putText(frame, f"out5(Grooming): {float('%.2f' % out5)}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            videoWriter.write(frame)
    else:
        videoWriter.release()
        break