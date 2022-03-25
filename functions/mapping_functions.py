import copy
import glob

import cv2
import numpy as np
import pandas as pd

import koger_general_functions as kgf

try:
    import gdal
except ImportError:
    print("Warning: importing 'mapping_funtions.py' without having installed gdal.")
    print("Will throw error if using functions that require gdal.")

def get_uv(xyz_raw, pmatrix, offset):
    '''
    Project from coordinates in map xyz coordinates back onto original image

    xyz_raw: the map coordinates
    pmatrix: the pmatrix from pix4d for the given image points are projecting to
    offset: from pix4d
    '''
    xyz = xyz_raw - offset
    xyz1 = np.ones((4, 1))
    xyz1[:3] = np.expand_dims(xyz, 1)
    xyz_image = np.matmul(pmatrix, xyz1)
    uv = np.array([xyz_image[0] / xyz_image[2], xyz_image[1] / xyz_image[2]])
    return np.squeeze(uv)


def get_xyz_proj(uv, inv_m, p4, mu, offset):
    '''
    Project from uv image coordinates to map xyz coordinates for some given mu

    uv: image coordinates
    inv_m: inverse m matrix (m matrix is the left three columns of pmatrix)
    p4: the fourth colummn of the pmtatix
    mu: a term that chooses a point along projection ray
    offset: from pix4d
    '''
    uv1 = np.ones(3)
    uv1[:2] = uv 
    xyz = np.matmul(inv_m, ((uv1 * mu) - p4))
    xyz += offset
    return xyz


def utm_to_raster(x_utm, y_utm, x_origin, y_origin, pixel_width, pixel_height, image_scale):
    '''
    Transforms from utm map coordinates to raster map coordinates

    x_utm: x coordinate in utm
    y_utm: y coordinate in utm
    x_origin: of raster image in utm 
    y_origin: of raster image in utm
    pixel_height: of raster image in utm
    pixel_width: of raster image in utm
    image_scale: if the raster image being used has been scaled from the original image
                 image scale of .5 means that the map is being used at
                 .5 w .5 h compared to original

    '''
    if np.isnan(x_utm) or np.isnan(y_utm):
        return((np.nan, np.nan))
    x_raster = int(((x_utm - x_origin) / pixel_width) * image_scale)  
    y_raster = int(((y_utm - y_origin) / pixel_height) * image_scale)
    return((x_raster, y_raster))


def utm_to_raster_track(track_utm, x_origin, y_origin, pixel_width, pixel_height, image_scale):

    '''
    Transforms from utm map coordinates to raster map coordinates for a given track

    track_utm: shape(n,2) track
    x_origin: of raster image in utm 
    y_origin: of raster image in utm
    pixel_height: of raster image in utm
    pixel_width: of raster image in utm
    image_scale: if the raster image being used has been scaled from the original image
                 image scale of .5 means that the map is being used at
                 .5 w .5 h compared to original

    '''
    track_raster = np.copy(track_utm)
    track_raster[:,0] = ((track_raster[:,0] - x_origin) / pixel_width) * image_scale 
    track_raster[:,1] = ((track_raster[:,1] - y_origin) / pixel_height) * image_scale 
    track_raster[~np.isnan(track_raster)] = track_raster[~np.isnan(track_raster)].astype(int)
    return track_raster



def raster_to_utm(x_raster, y_raster, x_origin, y_origin, pixel_width, pixel_height):
    '''
    opposite of utm_to_raster
    '''
    x_utm = ((x_raster * pixel_width) + x_origin)
    y_utm = (y_raster * pixel_height) + y_origin
    return((x_utm, y_utm))


def get_image_ind_from_frame_num(frame_num, frame_names, first_frame, first_frame_in_obs_this_flight):
    '''
    returns frame index.  Is equivelent to index for that frame in position matrix

    frame_num: the raw frame num in video (60fps including takeoff etc.)
    frame_names: list of frames being used for tracking (30fps, just observation)
    first_frame: first frame in video of tracked observation

    '''
#     if frame_num % 2 != 0:
#         frame_num -= 1
    frame_ind = int((frame_num - first_frame) // 2)
#     print(frame_ind + first_frame_in_obs_this_flight, len(frame_names))
    # sort of hack, when frame right after observation ends is used for a map
    if frame_ind + first_frame_in_obs_this_flight == len(frame_names):
        frame_ind -= 1
        print('last frame of map is one frame beyond observation end')
    frame_file = frame_names[frame_ind + first_frame_in_obs_this_flight]
    
    return [frame_ind, frame_file]

# def image_crop_to_map(pixel_values, pixel_locations, mu, pmatrix_dict, offset, p):
#     '''
#     Project from a block of uv image coordinates to map xy coordinates for some given mu

#     crop: image coordinates
#     inv_m: inverse m matrix (m matrix is the left three columns of pmatrix)
#     p4: the fourth colummn of the pmtatix
#     mu: a term that chooses a point along projection ray
#     '''
    
    
#     for pixel_value, pixel_location in zip()
    
#     uv1 = np.ones(3)
#     uv1[:2] = uv 
#     xyz = np.matmul(inv_m, ((uv1 * mu) - p4))
#     xyz += offset
#     return xyz

def correct_pmatrix(raw_pmatrix_dict, movement_matrix, mu_est):
    total_y = movement_matrix[0, 2] * mu_est
    total_x = movement_matrix[1, 2] * mu_est
    
    pmatrix_dict = copy.deepcopy(raw_pmatrix_dict)
    
    pmatrix_dict['p4'][0] -= total_y
    pmatrix_dict['p4'][1] -= total_x
    
    return pmatrix_dict
    


def from_image_to_map(uv, z_guess, pmatrix_dict, offset, elevation_map, 
    max_guesses, correct_threshold, pixel_width, pixel_height, x_origin, y_origin, 
    object_height=1):
    
    '''
    The project image coordinate on to map as the intersection between the 
    projection ray from camera through image point and the ground.  Iteritively
    searches for this point

    uv: point in image
    z_guess: where to start looking for ground
    pmatrix_dict: must_contain inv_mmatrix (inverse of the first three columns of pmatrix)
                  and p4 (fourth column of pmatrix)  
    offset: from pix4d
    elevation_map: elevation raster plot of map area
    max_guesses: how many iterations to search along projection ray before returning estimate
    correct_threshold: if the distance between the point on the projection ray and the ground is 
                       within this threshold stop seraching and return point
    pixel_width: pixel width of pixels in elevation raster in utm units
    pixel_height: like pixel width
    x_origin: origin of elevation raster in utm units
    y_origin: like x_origin
    object_height: expected height of object above ground

    returns: xy coordinates in utm, if the point is found, the mu value for the guess, and the corresponding real elevation 
    '''

    last_pos_guess = 0
    last_neg_guess = None
    found_point = False
    first_search = True
    guess_count = 0
    animal_height = object_height
    
    orig_z_guess = z_guess
    
    while not found_point:
        xyz = get_xyz_proj(uv, pmatrix_dict['inv_mmatrix'], 
                           pmatrix_dict['p4'], z_guess, offset)
        x_rast, y_rast = utm_to_raster(xyz[0], xyz[1], x_origin, y_origin, 
                                       pixel_width, pixel_height, 
                                       image_scale=1.0)
        x_rast = min(x_rast, elevation_map.shape[1]-1)
        x_rast = max(x_rast, 0)
        y_rast = min(y_rast, elevation_map.shape[0]-1)
        y_rast = max(y_rast, 0)
        elevation = elevation_map[y_rast, x_rast]
        z_diff = xyz[2] + animal_height - elevation

        if guess_count > max_guesses:
            break
        if np.abs(z_diff) <= correct_threshold:
            found_point = True
            break
        if first_search and z_diff > 0:
            last_pos_guess = z_guess
            z_guess += z_guess
        elif first_search and z_diff < 0:
            first_search = False
            new_guess = (last_pos_guess + z_guess) / 2
            last_neg_guess = z_guess
            z_guess = new_guess
        elif z_diff > 0:
            new_guess = (last_neg_guess + z_guess) / 2
            last_pos_guess = z_guess
            z_guess = new_guess
        elif z_diff < 0:
            new_guess = (last_pos_guess + z_guess) / 2
            last_neg_guess = z_guess
            z_guess = new_guess
        guess_count += 1
    x_utm, y_utm = raster_to_utm(x_rast, y_rast, 
                                 x_origin, y_origin, 
                                 pixel_width, pixel_height)
    
    if not found_point:
        print('big change', orig_z_guess, z_guess)
    
    
    
#     print('uv:', uv, 'p4:', pmatrix_dict['p4'], x_utm, 'y:', y_utm, 'fp:', found_point, 'z:', z_guess, 'el:', elevation)
        
    return (x_utm, y_utm, found_point, z_guess, elevation)



def create_pmatrix_dicts(pmatrix_file, big_map=False):
    '''
    Takes a file generated by pix4d with all of the calculated pmatrix values
    generates a dictionary that contains:
    image_name: image_name of image that corresponds with the given pmatrix
    pmatrix: numpy pmatrix
    inv_mmatrix: inverse of first three columns of pmatrix
    p4: last column if pmatrix

    pmatrix_file: path to file generated by pix4d

    returns list of dictioanries 
    '''
    with open(pmatrix_file) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    pmatrix_list = []
    for line in content:
        split_line = line.split(' ')
        pmatrix = np.zeros((3, 4))
        for row in range(pmatrix.shape[0]):
            pmatrix[row, :] = split_line[(1 + row * 4):(1 + (row+1) * 4)]
        mmatrix = np.copy(pmatrix[:, :3])
        inv_mmatrix = np.linalg.pinv(mmatrix)
        p4 = np.copy(pmatrix[:, 3])
        pmatrix_list.append({'image_name': split_line[0], 'pmatrix': pmatrix, 'inv_mmatrix': inv_mmatrix, 'p4': p4})
    if big_map:
        pmatrix_list.sort(key=lambda pmatrix_dict: 
            (pmatrix_dict['image_name'].split('_')[-3], 
                int(pmatrix_dict['image_name'].split('_')[-2]), 
                int(pmatrix_dict['image_name'].split('.')[-2].split('_')[-1]))) 
    else:
        pmatrix_list.sort(key=lambda pmatrix_dict: 
            (pmatrix_dict['image_name'].split('_')[-4], 
                int(pmatrix_dict['image_name'].split('_')[-2]), 
                int(pmatrix_dict['image_name'].split('.')[-2].split('_')[-1]))) 

    return pmatrix_list

def get_groundtruth_obs_indexes(flight_logs, frame_folders_root, for_test=False, 
                                use_old_method=False):
    '''
    Return list of observation indexes that are ground truth images
    
    flight_logs: list of dataframes linking drones logs and video frame numbers
    frame_folders_root: path to folder containing all the frame 
        folders for observation
    for_test: boolean, True if there are test groundtruth frames 
        that should ignored
    use_old_method: boolean, True is should use old function to
        calculate this
    '''
    
    if use_old_method:
        gtruth_obs_indexes, gtruth_frame_names = get_groundtruth_obs_indexes_old(
            flight_logs, frame_folders_root, for_test=for_test
        )
        
        return gtruth_obs_indexes, gtruth_frame_names
    
    gt_names = []
    for fl in flight_logs:
        if for_test:
            gt_inds = fl['used_in_map'] & ~fl['test_frame']
        else:
            gt_inds = fl['used_in_map']
        gt_names.extend(fl.loc[gt_inds, 'image_name'].to_list())
        
    obs_files = kgf.get_observation_frame_files(frame_folders_root)

    used_gt_names = []
    gt_obs_inds = []
    last_gt_obs_ind = 0
    for gt_ind, name in enumerate(gt_names):
        for obs_ind, file in enumerate(obs_files[last_gt_obs_ind:]):
            found = False
            if name in file:
                gt_obs_inds.append(obs_ind + last_gt_obs_ind)
                used_gt_names.append(file)
                last_gt_obs_ind += obs_ind
                found = True
                break

        if not found:
            if gt_ind != len(gt_names)-1:
                frame_num = int(name.split('.')[-2].split('_')[-1])
                new_frame_num = frame_num - 2
                new_name = f"{name[:-9]}{new_frame_num:05d}.jpg"
                for obs_ind, file in enumerate(obs_files[last_gt_obs_ind:]):
                    found = False
                    if new_name in file:
                        gt_obs_inds.append(obs_ind + last_gt_obs_ind)
                        used_gt_names.append(file)
                        last_gt_obs_ind += obs_ind
                        found = True
                        print(f"gt not found. using {new_name} for {name}.")
                        break
                        
            else:
                print(f"gt not found. setting to last frame.")
                gt_obs_inds.append(len(obs_files)-1)
                used_gt_names.append(obs_files[-1])
            
            
    return gt_obs_inds, used_gt_names


# THIS SHOULD BE DELETED
def get_groundtruth_obs_indexes_old(flight_logs, frame_folders_root, is_obs088=False, alt=False, for_test=False):
    '''
    Return list of observation indexes that are ground truth images
    
    flight_logs: list of dataframes linking drones logs and video frame numbers
    frame_folders_root: path to folder containing all the frame folders for observation
    is_obs088: to deal with a special case, only set true if processing observation088
    for_test: boolean, True if there are test groundtruth frames that should ignored
    '''

    
    frame_names = kgf.get_observation_frame_files(frame_folders_root)
    
    if alt:
        gtruth_obs_indexes = []
        last_gt_obs_ind = 0
        for gt_frame_name in list(flight_logs[flight_ind].loc[flight_logs[flight_ind]['used_in_map'], 'frame_name']):
            for ind, obs_frame_path in enumerate(frame_names[last_gt_obs_ind:]):
                if gt_frame_name in obs_frame_path:
                    gtruth_obs_index
    
    flight_names_for_each_frame = [frame_name.rpartition('/')[0].rpartition('_')[0] for frame_name in frame_names]
    unique_flight_names = list(set(flight_names_for_each_frame))
    unique_flight_names.sort()
    frames_in_prev_flight = [0]
    for flight_name in unique_flight_names:
        frames_in_prev_flight.append(flight_names_for_each_frame.count(flight_name))
        
    
    
    
    # observation index for each ground truth image in map
    gtruth_obs_indexes = []
    gtruth_frame_names = []
    
    obs_count = 0
    total_possible = 0

    obs_frames_in_previous_flights = 0
    for flight_ind in range(len(flight_logs)):
        # Frames used to make the pix4d map
        
        if for_test:
            print('Running with testing gt frames...')
            gt_mask = flight_logs[flight_ind]['used_in_map'] & ~flight_logs[flight_ind]['test_frame']
        else:
            gt_mask = flight_logs[flight_ind]['used_in_map']
            
        gtruth_frames_this_flight = list(flight_logs[flight_ind].loc[gt_mask, 'frame_num'])
        
        total_possible += len(gtruth_frames_this_flight)
        
        first_frame_this_flight = list(flight_logs[flight_ind].loc[flight_logs[flight_ind]['observing'], 'frame_num'])[0]

        obs_frames_in_previous_flights += frames_in_prev_flight[flight_ind]

        for frame_num in gtruth_frames_this_flight:

            frame_ind, frame_name = get_image_ind_from_frame_num(
                frame_num, frame_names, first_frame_this_flight, obs_frames_in_previous_flights)
            gtruth_obs_indexes.append(frame_ind + obs_frames_in_previous_flights)
#             print(obs_count, total_possible)
#             print('--------')
            gtruth_frame_names.append(frame_name)
            obs_count += 1

    if is_obs088:
        print('warning deleting two frames to deal with obs088')
        gtruth_obs_indexes = gtruth_obs_indexes[:81] + gtruth_obs_indexes[83:]


    # Make sure all the frames are accounted for 
    assert len(frame_names) == sum(frames_in_prev_flight)

    return gtruth_obs_indexes, gtruth_frame_names

def get_image_map(geotif_image_file):
    '''
    read 3d (rgb) raster map from geotif file. return as np array
    
    geotif_image_file: string - geotif file 
    '''
    image_gtif = gdal.Open(geotif_image_file)
    bands = []
    for band_num in range(1, 4):
        srcband = image_gtif.GetRasterBand(band_num)
        a = srcband.ReadAsArray()
        bands.append(a)
    image_map = np.stack(bands, 2)
    return image_map


def _clean_flight_logs(flight_logs, pmatrix_list, verbose=True):
    """ Mark all instances where image not actually used in map.
    
    flight_logs: list of modified flight_logs with column 'used_in_map'
        dataframe describing drone sensor info and coresponding video 
        frame numbers
    pmatrix_list: list of dicts created by kmap.create_pmatrix_dicts()
    verbose: if True print how many images not used
    """
    
    num_ims_not_used = 0
    # filenames of all images with assosiated pmatrix
    used_names = [p['image_name'] for p in pmatrix_list]
    for fl_ind, flight_log in enumerate(flight_logs):
        used_in_map = flight_log['used_in_map'] == True
        for ind, gt_row in flight_log[used_in_map].iterrows():
            if not gt_row['image_name'] in used_names:
                flight_log.loc[ind, 'used_in_map'] = False
                flight_log.loc[ind, 'test_frame'] = False
                num_ims_not_used += 1
    if verbose:
        print(f"{num_ims_not_used} gt images not used in final map.") 
        
    return flight_logs

def get_cleaned_flight_logs(flight_logs_folder, pmatrix_list):
    """
    Load and return flight logs that reflect frames map actually used.
    
    flight_logs_folder: path to folder containing flight logs
    pmatrix_list: info about each groundtruth image and resulting pmatrix
    """
    
    dataframe_files = glob.glob(flight_logs_folder + '/flight_*.pkl')
    dataframe_files.sort()
    flight_logs = [pd.read_pickle(file) for file in dataframe_files]
    flight_logs = _clean_flight_logs(flight_logs, pmatrix_list) 
    
    return flight_logs
    
    

def get_image_map(geotif_image_file):
    """
    Load geotif raster and return raster dict of pixel info.
    
    geotif_image_file: full path to a rgb geotif raster
    """
    
    image_gtif = gdal.Open(geotif_image_file)
    
    bands = []
    for band_num in range(1, 4):
        srcband = image_gtif.GetRasterBand(band_num)
        a = srcband.ReadAsArray()
        bands.append(a)
    image_map = np.stack(bands, 2)
    
    raster_info = {}
    
    image_transform = image_gtif.GetGeoTransform()
    raster_info['x_origin'] = image_transform[0]
    raster_info['y_origin'] = image_transform[3]
    raster_info['pixel_width'] = image_transform[1]
    raster_info['pixel_height'] = image_transform[5]
    
    
    return image_map, raster_info

def get_dtm(dtm_file, shape):
    """
    Load dtm and resize to new shape.
    
    dtm_file: full path to dtm_file
    shape: shape of returned array
    """
    dtm_gtif = gdal.Open(dtm_file)
    dtm = dtm_gtif.GetRasterBand(1).ReadAsArray()
    dtm_ave = np.min(dtm[dtm != -10000])
    dtm = np.where(dtm == -10000, dtm_ave, dtm)
    dtm_big = cv2.resize(dtm, (shape[1], shape[0]),interpolation=cv2.INTER_CUBIC)
    return dtm_big

def load_map_offset(offset_file):
    """
    Load and return offset file from pix4d for map.
    
    offset_file: full path to offset file
    """
    
    with open(offset_file) as f:
        offset_raw = f.read()
    offset = np.array(offset_raw.split(' ')).astype(float)
    return offset


def load_drone_movements(drone_movement_folder):
    """
    Load drone movement matricies and number of inliers for each matrix.
    
    Each groundtruth segment has list of matricies for each frame in segment.
    
    drone_movement_folder: full path to folder containing movement matricies
                           inliers
                           
    return movement matricies and inliers
    
    each is list gt-segments long with each list entry containing one matrix for each frame
    """
    
    segment_movement_files_list = glob.glob(drone_movement_folder +
                                            '/drone_movement_segment_*.npy')
    segment_movement_files_list.sort()

    segment_inlier_files_list = glob.glob(drone_movement_folder + 
                                          '/inliers_segment_*.npy')
    segment_inlier_files_list.sort()
    # Read list of affine movement matricies for each frame in each segment 
    segment_movements_list = [np.load(file) for file in segment_movement_files_list]
    segment_inliers_list = [np.load(file) for file in segment_inlier_files_list]
    
    return segment_movements_list, segment_inliers_list


def get_groundtruth_camera_locations(camera_locations_file):
    """Get groundtruth camera locations from pix4d file.
    
    Args:
        camera_locations_file: pix4d file:
            path + _calibrated_external_camera_parameters_wgs84.txt
    """
    with open(camera_locations_file, 'r') as f:
        data = f.readlines()
    camera_info_names = data[0].split(' ')
    camera_locations_unsorted = pd.DataFrame(columns=camera_info_names)    
    for line in data[1:]:
        camera_info = line.split(' ')
        camera_dict = {}
        for info_ind in range(len(camera_info)):
            if info_ind == 0:
                camera_dict[camera_info_names[info_ind]] = camera_info[info_ind]
            else:
                camera_dict[camera_info_names[info_ind]] = float(camera_info[info_ind])
        camera_locations_unsorted = camera_locations_unsorted.append(camera_dict, ignore_index=True)

    # Sort the drone camera coordinates in standard image sort order
    image_names = list(camera_locations_unsorted['imageName'])
    sort_lambda = lambda name: (name.split('_')[-4], name.split('_')[-2], 
                                int(name.split('.')[-2].split('_')[-1]))
    sorted_image_names = sorted(image_names, key=sort_lambda)
    sort_index = [image_names.index(i) for i in sorted_image_names]
    camera_locations = camera_locations_unsorted.iloc[sort_index].copy()
    camera_lat_long = camera_locations.loc[:,['latitude', 'longitude']]
    camera_lat_long = camera_lat_long.reset_index(drop=True)
    return camera_lat_long

def load_flight_logs(flight_log_folder):
    """ Load all flight logs in given folder.
    
    Args:
        flight_log_folder: path to folder containing log files
    """
    
    dataframe_files = glob.glob(flight_log_folder + '/flight_*.pkl')
    dataframe_files.sort()
    flight_logs = [pd.read_pickle(file) for file in dataframe_files]
    return flight_logs


def get_groundtruth_image_files(flight_logs, frame_folders_root):
    """ Get the image files used to make map.
    Args:
        flight_logs: list of flightlog dataframes
        frame_folders_roots: path to folder that contains 
            folders of frames (observation... in raw footage)
    """
    gt_image_names = []
    for flight_log in flight_logs:
        image_names = list(flight_log.loc[flight_log['used_in_map'], 'image_name'])
        gt_image_names.extend(image_names)
    gt_image_files = []
    for frame_num, frame_name in enumerate(gt_image_names):
        flight_name = '_'.join(frame_name.split('_')[:5])
        image_file = '{}/{}/{}'.format(frame_folders_root, flight_name, frame_name)
        gt_image_files.append(image_file)
    return gt_image_files

def get_number_of_frames_in_prev_flights(frame_files):
    """ Create list of number of frames in previous observation flights.
    
    Args:
        frame_files: list of all frame_files used in observation
    """
    
    all_flight_names = [f.rpartition('/')[0].rpartition('_')[0] for f in frame_files]
    unique_flight_names = list(set(all_flight_names))
    unique_flight_names.sort()
    frames_in_prev_flight = [0]
    for flight_name in unique_flight_names:
        frames_in_prev_flight.append(all_flight_names.count(flight_name))
        
    return frames_in_prev_flight

def get_gt_segment_from_obs_ind(obs_ind, gt_obs_indexes, guess_segment_ind=0):
    """ Find the gt segment the observation index is part of.
    
    Args:
        obs_ind: the obervation index you're interested in
        gt_obs_indexs: the observation indexes of the groundtruth frames
        guess_segment_ind: which segment you think it might be in
            (to speed up search time)
    """
    
    def _in_segment(obs_ind, gt_obs_indexes, guess_segment_ind):
        if gt_obs_indexes[-1] == obs_ind:
            return True
        if guess_segment_ind >= len(gt_obs_indexes) - 1:
            return False
        in_segment = ((obs_ind >= gt_obs_indexes[guess_segment_ind])
                      and (obs_ind < gt_obs_indexes[guess_segment_ind+1]))
        return in_segment
    
    
    while(not _in_segment(obs_ind, gt_obs_indexes, guess_segment_ind)):
        if guess_segment_ind < 0:
            guess_segment_ind = 0
        if obs_ind > gt_obs_indexes[-1]:
            # This frame is beyond last ground truth
            return False
        guess_segment_ind += 1
        if guess_segment_ind >= len(gt_obs_indexes):
            guess_segment_ind = 0
            
    return guess_segment_ind

def get_segment_step(obs_ind, gt_obs_indexes, segment_ind):
    """ Find where given observation index fits within given gt segement.
    
    Args:
        obs_ind: the observation index
        gt_obs_indexes: list of the observation indexes of gt frames
        segment_ind: the index of the segment of interest
    """
    
    segment_step = obs_ind - gt_obs_indexes[segment_ind]
    return segment_step

def is_track_in_obs_ind(track, obs_ind):
    """ Return true is track is active in obs_ind.
    
    Args:
        track: track dict
        obs_ind: observation index
    """
    
    if track['first_frame'] > obs_ind or track['last_frame'] < obs_ind:
        return False
    return True


def get_track_position_at_obs_ind(track, obs_ind, image_shape):
    """Return uv track position in frame at obs_ind.
    
    Return False if track not active in obs_ind.
    
    Args:
        track: track_dict
        obs_ind: observation index
        image_shape: dimensions of drone frame"""
    
    if not is_track_in_obs_ind(track, obs_ind):
        return False
    
    rel_frame_num = obs_ind - track['first_frame']
    u = track['track'][rel_frame_num, 1]
    v = image_shape[0] - track['track'][rel_frame_num, 0]
    position_uv = np.array([u, v])
    
    if np.any(np.isnan(position_uv)):
        return False
    
    return position_uv
    
    
    
    

def get_obs_ind_utms(tracks, obs_ind, rotation_matrix, pmatrix_dict, 
                               offset, elevation_r, mu_est, max_guesses, 
                               correct_threshold, pixel_width, pixel_height,
                               x_origin, y_origin, image_shape, bias=None, 
                               object_height=1):
    """ Calculate utm positions for every track at obs_ind.
    
    Return num tracks by two array. nan if track isn't present
    
    Args:
        tracks: list of track dicts
        obs_ind: observation index
        rotation_matrix: describes how points should be rotated from ground truth
        pmatrix_dict: must_contain inv_mmatrix (inverse of the first three columns of pmatrix)
                      and p4 (fourth column of pmatrix)  
        offset: from pix4d
        elevation_r: elevation raster plot of map area
        mu_est: where to start looking for ground
        max_guesses: how many iterations to search along projection ray before returning estimate
        correct_threshold: if the distance between the point on the projection ray and the ground is 
                           within this threshold stop seraching and return point
        pixel_width: pixel width of pixels in elevation raster in utm units
        pixel_height: like pixel width
        x_origin: origin of elevation raster in utm units
        y_origin: like x_origin
        bias: how much to modify utm points (known error between segments for instance)
        object_height: expected height of object above ground
    """
    
    utms = []
        
    num_tracks_in_frame = 0
    new_mu_sum = 0
    
    if bias is None:
        bias = np.zeros((len(tracks), 2), dtype=float)
    
    for track_ind, track in enumerate(tracks):

        track_uv = get_track_position_at_obs_ind(track, obs_ind, image_shape)
            
        if track_uv is False:
            # Track isn't present in this frame so just add nan
            utms.append(np.array([np.nan, np.nan]))
            continue
                
        num_tracks_in_frame += 1

        track_uv_rot = np.matmul(rotation_matrix, track_uv)
        
        
            
        x_utm, y_utm, found_point, mu, elevation = from_image_to_map(track_uv_rot, 
                                                                          mu_est, 
                                                                          pmatrix_dict, 
                                                                          offset, elevation_r, 
                                                                          max_guesses, 
                                                                          correct_threshold, 
                                                                          pixel_width, 
                                                                          pixel_height, 
                                                                          x_origin, y_origin,
                                                                          object_height)
        
        utms.append(np.array([x_utm, y_utm]) + bias[track_ind])

        new_mu_sum += mu

    utms = np.vstack(utms)
    if num_tracks_in_frame != 0:
        new_mu = new_mu_sum / num_tracks_in_frame
    else:
        new_mu = 0

    return utms, new_mu

def utm_to_raster_for_step(utms, x_origin, y_origin, pixel_width, 
                           pixel_height, image_scale):
    """ Convert all points in a segment step from utm to raster coordinates.
    
    Returns number of utms by 2 array
    
    Args:
        utms: list or array of utm points (x, y)
        x_origin: of raster image in utm 
        y_origin: of raster image in utm
        pixel_height: of raster image in utm
        pixel_width: of raster image in utm
        image_scale: if the raster image being used has been scaled 
            from the original image image scale of .5 means that 
            the map is being used at .5 w .5 h compared to original
    """
    
    raster_points = []
    
    for utm in utms:
        if np.any(np.isnan(utm)):
            raster_points.append(np.array([np.nan, np.nan]))
            continue
        x_r, y_r = utm_to_raster(utm[0], utm[1], x_origin, y_origin, 
                                      pixel_width, pixel_height, 
                                      image_scale=1.0)
        raster_points.append(np.array([x_r, y_r]))
        
    raster_points = np.vstack(raster_points)
    
    return raster_points

def calculate_obs_ind_utms_in_segment(tracks, obs_ind, gt_index, segment_step,
                                    pmatrix_list, segment_movements_list,
                                    mu_est, offset, elevation_r, max_guesses,
                                    correct_threshold, pixel_width, pixel_height,
                                    x_origin, y_origin, image_shape, bias=None,
                                    object_height=1
                                   ):
    """ Find all utm positions for tracks at given gt segment.
    
    
    Args:
        tracks: list of track dicts
        obs_ind: observation index
        gt_idex: groundtruth index
        segment_step: segment ind
        pmatrix_list: list of pmatrix dicts
        segment_movements_list: arrays relating frame to groundtruth
        mu_est: where to start looking for ground
        offset: from pix4d
        elevation_r: elevation raster plot of map area
        max_guesses: how many iterations to search along projection ray 
            before returning estimate
        correct_threshold: if the distance between the point on the 
            projection ray and the ground is within this threshold 
            stop seraching and return point
        pixel_width: pixel width of pixels in elevation raster in utm units
        pixel_height: like pixel width
        x_origin: origin of elevation raster in utm units
        y_origin: like x_origin
        bias: how much to modify utm points (known error between segments for instance)
        object_height: expected height of object above ground
        
    """
    if gt_index >= len(segment_movements_list):
        # Last frame
        if gt_index < len(pmatrix_list):
            movement_matrix = np.eye(3)
        else:
            print("Error, gt_index too high, returning False, False")
            return False, False
    else:
        # Translation from last ground truth
        movement_matrix = segment_movements_list[gt_index][segment_step]
    raw_pmatrix_dict = pmatrix_list[gt_index]
    pmatrix_dict = correct_pmatrix(raw_pmatrix_dict, movement_matrix, mu_est)
    # WHY NOT MULTIPLIED BY LAST MU
    rotation_matrix = copy.copy(movement_matrix[:2, :2])
        
    utms, new_mu = get_obs_ind_utms(tracks, obs_ind, rotation_matrix,
                                    pmatrix_dict, offset, elevation_r, 
                                    mu_est, max_guesses, correct_threshold, 
                                    pixel_width, pixel_height, x_origin, 
                                    y_origin, image_shape, bias)
    return utms, new_mu

def calculate_utms_for_segment_step(tracks, obs_ind, gt_index, segment_step,
                                    pmatrix_list, segment_movements_list,
                                    mu_est, offset, elevation_r, max_guesses,
                                    correct_threshold, pixel_width, pixel_height,
                                    x_origin, y_origin, image_shape, bias=None,
                                    object_height=1
                                   ):
    """ Find all utm positions for tracks at given gt segment.
    
    
    Args:
        tracks: list of track dicts
        obs_ind: observation index
        gt_idex: groundtruth index
        segment_step: segment ind
        pmatrix_list: list of pmatrix dicts
        segment_movements_list: arrays relating frame to groundtruth
        mu_est: where to start looking for ground
        offset: from pix4d
        elevation_r: elevation raster plot of map area
        max_guesses: how many iterations to search along projection ray before returning estimate
        correct_threshold: if the distance between the point on the projection ray and the ground is 
                           within this threshold stop seraching and return point
        pixel_width: pixel width of pixels in elevation raster in utm units
        pixel_height: like pixel width
        x_origin: origin of elevation raster in utm units
        y_origin: like x_origin
        bias: how much to modify utm points (known error between segments for instance)
        object_height: expected height of object above ground
        
    """
    print(" use 'kmap.calculate_utms_at_obs_index' instead of 'calculate_utms_for_segment_step'")
    
    
    utms, new_mu = calculate_obs_ind_utms_in_segment(tracks, obs_ind, gt_index, segment_step,
                                    pmatrix_list, segment_movements_list,
                                    mu_est, offset, elevation_r, max_guesses,
                                    correct_threshold, pixel_width, pixel_height,
                                    x_origin, y_origin, image_shape, bias,
                                    object_height
                                   )
    return utms, new_mu


def calculate_total_segment_error(tracks, obs_ind, gt_index,
                                pmatrix_list, segment_movements_list,
                                mu_est, offset, elevation_r, max_guesses,
                                correct_threshold, pixel_width, pixel_height,
                                x_origin, y_origin, image_shape, object_height=1):
    if gt_index + 1 == len(pmatrix_list):
        # no next segment
        return False
    
            
       # utms in last frame on current segment
    utms, new_mu = calculate_obs_ind_utms_in_segment(tracks, obs_ind, gt_index, -1,
                                                    pmatrix_list, segment_movements_list,
                                                    mu_est, offset, elevation_r, max_guesses,
                                                    correct_threshold, pixel_width, pixel_height,
                                                    x_origin, y_origin, image_shape, object_height=1
                                                   )
    utms, new_mu = calculate_obs_ind_utms_in_segment(tracks, obs_ind, gt_index, -1,
                                                    pmatrix_list, segment_movements_list,
                                                    new_mu, offset, elevation_r, max_guesses,
                                                    correct_threshold, pixel_width, pixel_height,
                                                    x_origin, y_origin, image_shape, object_height=1
                                                   )
    next_utms, new_mu = calculate_obs_ind_utms_in_segment(tracks, obs_ind, gt_index+1, 0,
                                                    pmatrix_list, segment_movements_list,
                                                    new_mu, offset, elevation_r, max_guesses,
                                                    correct_threshold, pixel_width, pixel_height,
                                                    x_origin, y_origin, image_shape, object_height=1
                                                   )
    
    error = next_utms - utms
    return(error)


def calculate_bias_for_segment(tracks, obs_ind, gt_index,
                                pmatrix_list, segment_movements_list,
                                mu_est, offset, elevation_r, max_guesses,
                                correct_threshold, pixel_width, pixel_height,
                                x_origin, y_origin, image_shape, object_height=1):
    
    """ Calculate difference between positions in last segment and new segment.
    
    Return difference divided by number of steps in segment.
    
    
    Args:
        tracks: list of track dicts
        obs_ind: observation index
        gt_index: groundtruth index
        pmatrix_list: list of pmatrix dicts
        segment_movements_list: arrays relating frame to groundtruth
        mu_est: where to start looking for ground
        offset: from pix4d
        elevation_r: elevation raster plot of map area
        max_guesses: how many iterations to search along projection ray before r
            eturning estimate
        correct_threshold: if the distance between the point on the projection ray 
            and the ground is within this threshold stop searching and return point
        pixel_width: pixel width of pixels in elevation raster in utm units
        pixel_height: like pixel width
        x_origin: origin of elevation raster in utm units
        y_origin: like x_origin
        image_shape: (height, width) of drone frame
        object_height: expected height of object above ground
        
    """
    
    
    if gt_index + 1 >= len(segment_movements_list):
        # no next segment
        return np.zeros((len(tracks), 2), dtype=float)
    
    error = calculate_total_segment_error(tracks, obs_ind, gt_index,
                                pmatrix_list, segment_movements_list,
                                mu_est, offset, elevation_r, max_guesses,
                                correct_threshold, pixel_width, pixel_height,
                                x_origin, y_origin, image_shape, object_height=1)
    error[np.isnan(error)] = 0.0
    bias = error / len(segment_movements_list[gt_index])
    
    return bias

def fill_with_min(array, value=-10000):
    """Fill in all unspecified values in array with minimum specified value.
    Assumes all unspecified values are set to constant value.
    
    arrary: np.array
    value: the value in array that corresponds with uspecified value
    """
    
    min_val = np.min(array[array!= value])
    array = np.where(array == value, min_val, array)
    return array

               
        
    
                      
                      
        
    

    
    
        
    
    
    