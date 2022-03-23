import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os


import sys
# To see koger_general_functions
sys.path.append('/home/golden/coding/drone-tracking/code/functions')
import koger_general_functions as kgf
import mapping_functions as kmap

def get_warp(frame0_gray, frame1_gray, plot_inliers=True):
    """ Return the calculated warp from frame1 to frame0."""
    
    max_number_of_tries = 3
    feature_quality_thresh = 0.2
    feature_thresh = 50
    
    features = get_image_keypoints(frame0_gray, max_number_of_tries, 
                                   feature_quality_thresh, feature_thresh
                                  )
    warp, num_inliers = get_movement_matrix(frame0_gray, frame1_gray, 
                                            features, plot_inliers=plot_inliers
                                           )
    
    return warp, num_inliers


def get_movement_matrix(im0_gray, im1_gray, good_points_im0,
                        plot_inliers=False):
    """
    Get homography from grayscale image 1 to grayscale image 0
    
    im0_gray: 2D matrix
    im1_gray: 2D matrix
    good_points_im0: output from cv2.goodFeaturesToTrack() for im0_gray
    plot: if True, plot points used for transform on both images
    """
    
    
    points1, status, error = cv2.calcOpticalFlowPyrLK(im0_gray, im1_gray, 
                                                      good_points_im0, None
                                                     )
    good_idx = np.where(status==1)[0]
    good_points0 = good_points_im0[good_idx]
    good_points1 = points1[good_idx]

    transform, inliers = cv2.estimateAffinePartial2D(good_points1, 
                                                     good_points0, 
                                                     ransacReprojThreshold=10
                                                    )
    warp_matrix = np.vstack((transform, np.array([0,0,1])))
    num_inliers = len(np.where(inliers==1)[0])
    
    if plot_inliers:
        good_ind = np.where(inliers==1)[0]

        plt.figure(figsize=(20,20))
        plt.imshow(im0_gray, cmap='gray')
        for circle in good_points0[good_ind]:
            plt.scatter(circle[0][0],circle[0][1], s=10)

        plt.figure(figsize=(20,20))
        plt.imshow(im1_gray, cmap='gray')
        for circle in good_points1[good_ind]:
            plt.scatter(circle[0][0],circle[0][1], s=10)


    return warp_matrix, num_inliers

def get_image_keypoints(frame, max_tries=3,
                        feature_quality_thresh=0.2, 
                        feature_thresh=50):
    """ Get good keypoints for movement tracking.
        
    Args:
        frame: grayscale image
        max_tries: how many times to reduce quality thresh if not enough points
        feature_quality_thresh: threshold for image keypoint features
        feature_thresh: min number of features features to stop looking
    """
    number_tries = 0
        
    potential_features = cv2.goodFeaturesToTrack(frame, maxCorners=300, 
                                                 qualityLevel=feature_quality_thresh, 
                                                 minDistance=50, blockSize=3)
    while (len(potential_features) < feature_thresh) and (number_tries < max_tries):
        feature_quality_thresh /= 2
        potential_features = cv2.goodFeaturesToTrack(frame, maxCorners=300, 
                                                     qualityLevel=feature_quality_thresh, 
                                                     minDistance=50, blockSize=3)
        number_tries += 1
    if len(potential_features) < feature_thresh:
         print('failed to reach feature threshold... ({})'.format(len(potential_features)))

    return potential_features

def get_pseudo_gt_seg_inds(segment_im_files, num_pseudo_gts):
    """ Get the segment indexes of the pseudo groundtruth frames
            based on number of frames and number of pseudo gts.
            
        Args:
            segment_im_files: frame files in gt segment
            num_pseudo_gts: how many pseudo ground truths if enough frames
    """
    num_frames_in_seg = len(segment_im_files)
    if num_frames_in_seg < num_pseudo_gts * 2:
        num_pseudo_gts = num_frames_in_seg // 2
        if num_pseudo_gts <= 0:
            num_pseudo_gts = 0
    
    pseudo_gt_frame_inds = np.linspace(0, num_frames_in_seg, 
                                       num=num_pseudo_gts, 
                                       endpoint=False, dtype=np.int
                                      )
    return pseudo_gt_frame_inds

def get_segment_drone_movement(segment_dict):
    """
    Get drone movement for every frame between two ground truth frames. 
        Use nine additional pseudo gts spaced throughout segment
    
    Assumes first and last image files in segement are ground truth images
    
    segement_dict: dictionary with following keys 
        'segment_im_files': frame files 
        'output_folder': path of save location
        'segment_number': segment number in observation
        'save': whether should save movement files
    """
    
    segment_im_files = segment_dict['segment_im_files']
    output_folder = segment_dict['output_folder']
    segment_number = segment_dict['segment_number']
    save = segment_dict['save']
    
    max_number_of_tries = 3
    feature_quality_thresh = 0.2
    feature_thresh = 50
    
    num_pseudo_gts = 10
    
    pseudo_gt_seg_inds = get_pseudo_gt_seg_inds(segment_im_files, num_pseudo_gts)
    
    pseudo_gt_frames = []
    for frame_ind in pseudo_gt_seg_inds:
        raw_im = cv2.imread(segment_im_files[frame_ind])
        gray_im = cv2.cvtColor(raw_im, cv2.COLOR_BGR2GRAY)
        pseudo_gt_frames.append(gray_im)
    
    pseudo_gt_frame_features = []
    for frame in pseudo_gt_frames:
        potential_features = get_image_keypoints(frame, max_number_of_tries,
                                                 feature_quality_thresh, 
                                                 feature_thresh
                                                )
        pseudo_gt_frame_features.append(potential_features)
        
    segment_transforms = []
    used_pgt_seg_inds = []
    num_inliers_list = []
    
    # the transformation nessisary to get from current pseudo gt
    # to the sements first gt image
    base_transform = np.eye(3)

    pseudo_gt_ind = 0
    pseudo_gt_seg_ind = 0 # Segment index of current pseudo gt
    
    pseudo_gt_frame = pseudo_gt_frames[pseudo_gt_ind]
    pseudo_gt_features = pseudo_gt_frame_features[pseudo_gt_ind]
    for file_num, im_focal_file in enumerate(segment_im_files[:]):
        im_focal = cv2.imread(im_focal_file)
        im_focal_gray = cv2.cvtColor(im_focal, cv2.COLOR_BGR2GRAY)
        
        warp, num_inliers = get_movement_matrix(pseudo_gt_frame, 
                                                im_focal_gray, 
                                                pseudo_gt_features
                                               )
        if num_inliers < feature_thresh:
            # add a pseudo gt using last frame
            base_transform = segment_transforms[-1]
            pseudo_gt_seg_ind = file_num - 1
            new_pgt_im = cv2.imread(segment_im_files[file_num-1])
            pseudo_gt_frame = cv2.cvtColor(new_pgt_im, cv2.COLOR_BGR2GRAY)
            pseudo_gt_features = get_image_keypoints(pseudo_gt_frame, 
                                                     max_number_of_tries,
                                                     feature_quality_thresh, 
                                                     feature_thresh
                                                    )
            warp, num_inliers = get_movement_matrix(pseudo_gt_frame, 
                                                im_focal_gray, 
                                                pseudo_gt_features
                                               )
            if num_inliers < feature_thresh:
                print(f"Warning. Warp features below threshold: {num_inliers}, seg num: {segment_number}.")

        warp = np.matmul(base_transform, warp)
        
        segment_transforms.append(warp)
        num_inliers_list.append(num_inliers)
        used_pgt_seg_inds.append(pseudo_gt_seg_ind)
        
        if pseudo_gt_ind < len(pseudo_gt_seg_inds) - 1:
            # Need to look ahead to see if should move to next ground truth
            if file_num >= pseudo_gt_seg_inds[pseudo_gt_ind + 1]:
                # This file number corresponds to next pseudo gt
                pseudo_gt_ind += 1
                base_transform = warp
                pseudo_gt_seg_ind = file_num
                pseudo_gt_frame = pseudo_gt_frames[pseudo_gt_ind]
                pseudo_gt_features = pseudo_gt_frame_features[pseudo_gt_ind]

    if save:
        segments_file = os.path.join(output_folder, 
                                     f"drone_movement_segment_{segment_number:03d}"
                                    )
        np.save(segments_file, segment_transforms)
        inliers_file = os.path.join(output_folder, 
                                    f"inliers_segment_{segment_number:03d}"
                                   )
        np.save(inliers_file, num_inliers_list)
        
    
    return segment_transforms, num_inliers_list, used_pgt_seg_inds

def create_gt_segment_dicts(frame_folders_root, flight_logs, 
                            output_folder=None, save=False):
    """Create list of dicts where each dict contains all image files for that gt segment.
    
    Args:
        frame_folders_root: folder containing the observations frames
        flight_logs: clean flight logs for observation
        output_folder: where to save transform info when segment is processed
        save: when processed should transform info be saved
    """
    
    
    if output_folder:
        try:
            os.makedirs(output_folder)    
        except FileExistsError:
            print('Warning, {} already exists'.format(output_folder))
    
    gtruth_obs_indexes, _ = kmap.get_groundtruth_obs_indexes(flight_logs, 
                                                             frame_folders_root
                                                            )
    obs_image_files = kgf.get_observation_frame_files(frame_folders_root)

    segment_dicts = []
    # -1 because we want segments between grountruth points (so n-1 segments for n grountruths)
    for gtruth_index, gtruth_obs_index in enumerate(gtruth_obs_indexes[:-1]):
        last_seg_ind = gtruth_obs_indexes[gtruth_index+1]
        segment_im_files = obs_image_files[gtruth_obs_index:last_seg_ind+1]
        segment_dicts.append({'segment_im_files': segment_im_files, 
                              'output_folder': output_folder, 
                              'segment_number': gtruth_index, 
                              'save': save})
    
    return segment_dicts

def process_new_anchor(im_file):
    """ Extract and return image and keypoint image for new anchor frame.
    
    Args:
        im_file: full path to image file
    
    Returns grayscale image and keypoint info.
    """
    raw_im = cv2.imread(im_file)
    gray_im = cv2.cvtColor(raw_im, cv2.COLOR_BGR2GRAY)
    
    image_features = get_image_keypoints(gray_im)
    
    return gray_im, image_features

def get_anchor_frames_without_log(frame_files, inlier_threshold=100,
                                 max_pseudo_anchors=7, verbose=True):
    """ Choose anchor frames to use for SfM mapping.
    
    Chooses anchor frames based on quality of local movement estimates (quantified
    by number of valid features to lst anchor) instead of from drone log information.
    
    Args:
        frame_files: list of full paths to all frames in observation (sorted)
        inlier_threshold: minimum number of points used to calculate warp
            to last (pseudo) anchor before start using next (pseudo) anchor
        max_pseudo_anchors: how many pseudo anchors to use before adding new
            anchor. (To ensure good results, should be less than 10 because 
            that is the default for the actual drone movement calculation)
            
    Return list of anchor frame files and warps between them (to check quality)
        """

    anchor_files = []
    final_warps = []

    anchor_files.append(frame_files[0])
    # The frame and features of current (pseudo) anchor frame
    a_frame, a_features = process_new_anchor(frame_files[0])
    # Number of pseudo anchors already used in this anchor segment
    num_pseudo_anchors = 0
    # Warp used for the previous frame (save this one, not the warp assosiated
    # with the first frame that doesn't have enough features)
    last_warp = None
    # Warp to get from current pseudo anchor frame back to the anchor frame
    base_transform = np.eye(3)

    for file_num, im_focal_file in enumerate(frame_files):
        if verbose:
            if file_num % 2000 == 0:
                print(f"{file_num} of {len(frame_files)} frames processed.",
                      f" {len(anchor_files)} anchors saved.")
        im_focal = cv2.imread(im_focal_file)
        im_focal_gray = cv2.cvtColor(im_focal, cv2.COLOR_BGR2GRAY)
        warp, num_inliers = get_movement_matrix(a_frame, im_focal_gray, a_features)
        warp = np.matmul(base_transform, warp)
        if num_inliers < inlier_threshold:
            a_frame, a_features = process_new_anchor(frame_files[file_num-1])
            if num_pseudo_anchors < max_pseudo_anchors:
                num_pseudo_anchors += 1
                base_transform = last_warp
            else:
                anchor_files.append(frame_files[file_num-1])
                final_warps.append(np.copy(last_warp))
                base_transform = np.eye(3)
                num_pseudo_anchors = 0
            continue
        last_warp = warp
    # Add last frame no matter what
    anchor_files.append(frame_files[-1])
    final_warps.append(np.eye(3))
    
    return anchor_files, final_warps