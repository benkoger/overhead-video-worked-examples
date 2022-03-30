import cv2
import numpy as np
import os

import koger_general_functions as kgf

def get_padded_crop(frame, center, crop_size):
    """Crop square around center in frame with zero padding where nessisary.

    Note: crop_size must be even.

    """
    if crop_size % 2 == 1:
        raise ValueError("Only even crop sizes excepted")
    center = center.astype(int)
    box = [[] for _ in range(4)]
    box_padding = [None for _ in range(4)]
    box[0] = int(frame.shape[0] - center[0] - crop_size / 2)
    if box[0] < 0:
        box_padding[0] = -box[0]
        box[0] = 0
    box[1] = int(center[1] - crop_size / 2)
    if box[1] < 0:
        box_padding[1] = -box[1]
        box[1] = 0
    box[2] = int(frame.shape[0] - center[0] + crop_size / 2)
    if box[2] > frame.shape[0]:
        box_padding[2] = frame.shape[0] - box[2]
        box[2] = frame.shape[0]
    box[3] = int(center[1] + crop_size / 2)
    if box[3] > frame.shape[1]:
        box_padding[3] = frame.shape[1] - box[3]
        box[3] = frame.shape[1]
    crop = frame[box[0]:box[2], box[1]:box[3]]
    if not all(padding is None for padding in box_padding):
        padded_crop = np.zeros((int(crop_size), int(crop_size), 3)).astype(np.uint8)
        padded_crop[box_padding[0]:box_padding[2], box_padding[1]:box_padding[3]] = crop
        crop = padded_crop
    assert crop.shape[0] == crop_size
    assert crop.shape[1] == crop_size
    return crop

def extract_crops(crop_info, return_crops=False):
    """ Extract and save crop at specified locations in specified frame.

    Pad black if position is close to edge of the frame.

    Args:
        crop_info: dict containing:
            'frame_file': path to image file
            'centers': list of positons in frame (same coordinates as recorded in track)
            'track_nums': track numbers in frame in same order as centers
            'obs_ind': the observation index (frame number in simple case)
            'crop_size': length and width of crop
            'save_folders': path to folders to save crop, if None don't save
        return_crops: If True, return crops

    return crops extracted from the frame
    """

    frame = cv2.imread(crop_info['frame_file'])
    centers = crop_info['centers']
    track_nums = crop_info['track_nums']
    save_folders = crop_info['save_folders']
    crops = []
    for center, track_num, save_folder in zip(centers, track_nums, save_folders):
        if np.any(np.isnan(center)):
            crop = np.zeros((crop_size, crop_size, 3))
        else:
            crop = get_padded_crop(frame, center, crop_info['crop_size'])

        if save_folder:
            filename = f"frame{track_num:02d}_{crop_info['obs_ind']:05d}.png"
            cv2.imwrite(os.path.join(save_folder, filename), crop)
        crops.append(crop)
    if return_crops:
        return crops


def create_observation_crop_dicts(frame_files, tracks_file, crop_size, save_folder,
                                  crops_per_subfolder=None):
    """ Create the 'crop_info' dicts used by extract_crop for all tracks in observation.

    Args:
        frame_files: sorted list of full paths to all frames in observation
        tracks_file: full path to file containing list of tracks in observation
        crop_size: size of crops to extract from frame
        save_folder: where to save crops (within this folder each track will have
            a subfolder. Each track folder may also have multiple subfolders so
            a single folder isn't overwhelmed with files for long tracks which
            could have hundred thousand+ crops)
        crops_per_subfolder: how many crops to store in a subfolder within a track
            before creating a new one. If None, store all crops in single sub
            folder
        """

    tracks_list = np.load(tracks_file, allow_pickle=True)

    crop_dicts = []

    for obs_ind, frame_file in enumerate(frame_files):
        positions = [] # current positions of tracks in frame
        track_nums = [] # track number of tracks in frame
        save_folders = [] # folder in which each crop should be saved
        for track_num, track in enumerate(tracks_list):
            track_ind = kgf.get_tracks_rel_frame_num(track, obs_ind)
            if track_ind:
                positions.append(track['track'][track_ind])
                track_nums.append(track_num)
                if not save_folder:
                    save_folders.append(None)
                    continue
                if crops_per_subfolder:
                    subfolder = track_ind // crops_per_subfolder
                else:
                    subfolder = 0
                crop_save_folder = os.path.join(
                    save_folder, f"track-{track_num}", f"segment-{subfolder}")
                os.makedirs(crop_save_folder, exist_ok=True)
                save_folders.append(crop_save_folder)

        if len(positions) == 0:
            continue

        crop_dict = {'frame_file': frame_file,
                     'centers': positions,
                     'track_nums': track_nums,
                     'obs_ind': obs_ind,
                     'crop_size': crop_size,
                     'save_folders': save_folders
                    }
        crop_dicts.append(crop_dict)

    return crop_dicts
