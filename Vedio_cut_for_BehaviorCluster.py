'''
Programming for cutting behavior moving
'''

# Import kits
import cv2
import os
import numpy
import pandas as pd
from moviepy.editor import *
import numpy as np
from CNN_for_shapes_data import resort_file
from time import sleep

if __name__ == "__main__":
    # generate a workspace for video cutting
    WorkingFolder = r'D:\project\BCI\data\data_30\20230520FOV\Behavior'  # setting work-folder
    VideoPath = f'{WorkingFolder}/video'  # define cutting space video
    os.makedirs(f'{VideoPath}/merged')

    # select mp4 file in video
    FileList = os.listdir(VideoPath)
    Mp4_file = []
    for text in FileList:
        if "mp4" in text:
            Mp4_file.append(text)

    video = VideoFileClip(filename=f'{VideoPath}/{Mp4_file[0]}')   # read video

    # check time bins for video cutting
    clusters = pd.read_csv(f'{WorkingFolder}/Full_umap.csv')   # read csv
    clusters.rename(columns={'Unnamed: 0': 'rank'}, inplace=True)

    # this part for for...
    for clu in range(1,max(clusters['labels']+1),1):
        print(clu)
        clusters_tmp = clusters[['rank', 'labels', 'time']][clusters["labels"] == clu]  # select cluster
        CutPath = f'{WorkingFolder}/video/cluster_{clu}'  # define cutting space video
        os.makedirs(CutPath)
        # generate a new data.frame
        new_index = range(0, (clusters_tmp.shape[0]), 1)
        dictionary = dict(zip(clusters_tmp.index, new_index))
        clusters_tmp.rename(index=dictionary, inplace=True)

        # This module using to cut continuous time blocks
        start_time = []
        end_time = []  # create number list for pandas

        # get start points & end points
        for i in range(0, (clusters_tmp.shape[0] - 1), 1):
            time_0 = clusters_tmp.loc[i, 'time']
            time_1 = clusters_tmp.loc[i + 1, 'time']
            diff_rank = time_1 - time_0

            if diff_rank > 0.06:
                end_time.append(time_0)
                start_time.append(time_1)

        start_time.insert(0, clusters_tmp.loc[0, 'time'])
        end_time.append(clusters_tmp.loc[(clusters_tmp.shape[0] - 1), 'time'])

        # Transform list to array
        start_time = numpy.array(start_time)
        end_time = numpy.array(end_time)
        diff_time = end_time - start_time

        # select
        fragments = (diff_time > 1)
        start_time = start_time[fragments]
        end_time = end_time[fragments]

        # this module for cutting video
        for i in range(0, (len(start_time))):
            video_new = video.subclip(start_time[i], end_time[i])
            video_new.to_videofile(f'{CutPath}/cluster{clu}_{i}.mp4', fps=30, remove_temp=False)

        # merge all fragments into a video
        video_l = []
        FileList = os.listdir(CutPath)
        FileList = resort_file(FileList=FileList)
        for file in FileList:
            filePath = f'{CutPath}/{file}'
            video_tmp = VideoFileClip(filePath)
            video_l.append(video_tmp)


        # merging video
        video_out = concatenate_videoclips(video_l)
        # generating targe
        video_out.to_videofile(f'{VideoPath}/merged/cluster{clu}.mp4', fps=30, remove_temp=True)

        print(f"cluster{clu} is finished")







