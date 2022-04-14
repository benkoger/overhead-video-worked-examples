'''
This is a set of functions that are useful in various parts of the animal tracking processing pipeline

'''

import os
import glob

import pandas as pd


# VIDEO/FRAME PROCESSING


def get_observation_frame_files(raw_frame_folder):

    """ Return list of sorted file names of each video frame in observation

    raw_frame_folder -- full path to folder containing one folder for each video clip in observation
    each clip folder contains all the frames in that clip.  The folder also contains a 
    .csv file that has the same name as the folder. This csv file has the first and last (exclusive)
    frame that is used in the observation for each video clip. 

    Assumes file name of form *_first-sort_whatever_second-sort_last-sort.jpg

    Assumes the frames in the clip folder we are every other frame in the raw video (ie video 60fps but folder has frames for 30fps)
    """

    csv_file = os.path.join(raw_frame_folder, os.path.basename(raw_frame_folder) + '.csv')
    try:
        observation_df = pd.read_csv(csv_file)
    except IOError:
        print('.csv file missing from folder')
        return False

    filenames = []
    for row in range(observation_df.shape[0]):
        # sample the video at every other frame
        # csv records raw frame numbers (say 60fps)
        first_index = int(observation_df.loc[row, 'first_frame'] / 2)
        last_index = observation_df.loc[row, 'last_frame']
        # ignore clips not included in observation at all
        if last_index == 0:
            continue
        
        frames = glob.glob(
            os.path.join(raw_frame_folder, observation_df.loc[row, 'video_name'], "*.jpg")
        )
        frames.sort()
            
        if last_index == -1:
            # We want to include the very last frame (so can't do :-1)
            filenames.extend(frames[first_index:])
        else:
            last_index = int(last_index / 2)
            filenames.extend(frames[first_index:last_index])

    filenames.sort(key=lambda file: (file.split('_')[-4], file.split('_')[-2], int(file.split('.')[-2].split('_')[-1])))

    return filenames

# TRACKS

def get_tracks_rel_frame_num(track, frame_num):
    """
    Return step index for particular track based on observation frame number. Return False if not in frame.
    
    track: a track dictionary
    frame_num: frame number in observation
    """
    
    if track['first_frame'] > frame_num or track['last_frame'] < frame_num:
        # track is not in current frame
        return False
    
    rel_frame_num = frame_num - track['first_frame']
    
    return rel_frame_num


