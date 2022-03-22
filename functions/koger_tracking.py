import numpy as np

def _get_boxes_center(boxes, frame_width=None, frame_height=None):
    # Need frame_width and frame_height if boxes are scaled 0 to 1 and
    # want output to be in frame coordinates
    center = np.ones((boxes.shape[0], 2))
    if frame_width and frame_height:
        #need to convert from top right to bottom right origin
        center[:, 0] = (frame_height - 
                        (frame_height * (boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2.0))) 
        center[:, 1] = frame_width * (boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2.0)
    else:
        center[:, 0] = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2.0
        center[:, 1] = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2.0
    return center

def normalize(vec):

    if len(vec.shape) == 2:
        vec_mag = np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)
    else:
        vec_mag = np.squeeze(np.sqrt(vec[0] ** 2 + vec[1] ** 2))
    return(np.expand_dims(vec / vec_mag, 0))

def add_all_points_as_new_tracks(raw_track_list, current_frame_ind, positions, 
                                 class_labels=None, noise=10, contours=None, 
                                 sizes=None):
    """ When there are no active tracks, add all new points to new tracks.
    
    Args:
        raw_track_list (list): list of tracks
        current_frame_ind (int): current frame index
        positions (numpy array): points x 2
        class_labels: if multiclass detection
        noise: Sometimes a false point will momentarily pop up. 
                Treating this noise as a real track causes problems because it 
                constrains the search space of a real nearby tracks. Instead
                we only treat a track as a real track if it's alreadybeen added 
                to.
        contours (list): segmentation contour for each point
        sizes: if points have size values
        
        noise: how much noise to add to tracks initially
    """
    for ind, position in enumerate(positions):
        if class_labels is not None:
            class_label = class_labels[ind]
        else:
            class_label = None
        if contours is not None:
            contour = contours[ind]
        else:
            contour = None
        if sizes is not None:
            size = sizes[ind]
        else:
            size = None
        raw_track_list.append(
            create_new_track(first_frame=current_frame_ind, 
                             first_position=position, pos_index=ind, 
                             class_label=class_label, noise=noise, 
                             contour=contour, size=size
                            )
        )
    return raw_track_list


def create_new_track(first_frame, first_position, pos_index, class_label=None, 
                     head_position=None, noise=10, contour=None, size=None,
                     debug=False):
    """ Create the dictionary which discribes a new track.
        
        Args:
            first_frame: first frame track appears
            last_frame: frame in which it was last seen
            pos_index: detection index in frame
            class_label: if multiclass detection
            noise: Sometimes a false point will momentarily pop up. 
                Treating this noise as a real track causes problems because it 
                constrains the search space of a real nearby tracks. Instead
                we only treat a track as a real track if it's alreadybeen added 
                to.
            debug: Add list to track where track creation events can be recorded
    """
    new_track = {'track': [first_position],
                'first_frame': first_frame,
                'last_frame': first_frame, 
                'pos_index': [pos_index],
                'noise': noise
                }
    
    if class_label is not None:
        new_track['class'] = [class_label]
    if contour is not None:
        new_track['contour'] = [contour]
    if size is not None:
        new_track['size'] = [size]
    if debug is not None:
        new_track['debug'] = ['{} created'.format(first_frame)]
    
    return new_track

def update_unmatched_track(track, debug=False):
    ''' Update track for next frame when there is no new point to add.
    
        Basically add a bunch of nans and point stays put. If
        track has positive noise value then noise value increases.
        
        Args:
            tracks: track dictionary
    '''
    
    track['track'].append(track['track'][-1])
    track['pos_index'].append(np.nan)
    if 'size' in track:
        track['size'].append(np.nan)
    if 'rects' in track:
        track['rects'].append(np.array([[np.nan, np.nan]]))
    if 'angles' in track:
        track['angles'].append(np.nan)
    if 'contour' in track:
        track['contour'].append(np.array([np.nan]))
    if track['noise'] > 0:
        # isn't a confirmed real track yet
        track['noise'] += 1
    if 'class_label' in track:
        track['class_label'].append(np.nan)
    if debug:
        track['debug'].append('Track wasn\'t matched.')
        
    return track

def update_matched_track(track, frame_index, new_position, 
                         new_position_index, size=np.nan, 
                         rect=[np.nan, np.nan], angle=np.nan,
                         contour=np.array(np.nan), class_label=np.nan):
    ''' Update track for next frame when there is a new point to add.
        
        Args:
            track: track dictionary
            frame_index (int): current observation frame
            new_position (np array): new position to add to track
            new_position_index (int): raw index of new point
            size (int): size of object being added
            rect (object): cv2 rect object 
            angle (float): angle of point being added
            contour (object): cv2 contour
            class_label(int): detected class label
    '''
    
    
    track['track'].append(new_position)
    track['pos_index'].append(new_position_index)
    track['last_frame'] = frame_index
    
    if track['noise'] > 0:
        # Maybe this should go to 0 after found
        track['noise'] -= 1
    if 'size' in track:
        track['size'].append(size)
    if 'rects' in track:
        track['rects'].append(rect)
    if 'angles' in track:
        track['angles'].append(angle)
    if 'contour' in track:
        track['contour'].append(contour)
    if 'class_label' in track:
        track['class_label'].append(class_label)

    return track

def _get_track_lengths(frame_ind, track_list, active_list):
    """ Calculate the current length of every active track.
    
    Args:
        frame_ind (num): current frame index
        track_list (list): list of all tracks
        active_list (list): list of all track indexes that are active
    
    return array of lengths of each track
    """
    
    track_lengths = np.zeros(len(active_list))
    for active_num, track_num in enumerate(active_list):
        track_length = frame_ind - track_list[track_num]['first_frame']
        track_lengths[active_num] = track_length
        
    return track_lengths

def _get_track_sizes(frame_ind, track_list, active_list):
    """ Calculate the current size of every active track.
    
    Args:
        frame_ind (num): current frame index
        track_list (list): list of all tracks
        active_list (list): list of all track indexes that are active
    
    return array of lengths of each track
    """
    
    track_sizes = np.zeros(len(active_list))
    for active_num, track_num in enumerate(active_list):
        track_size = track_list[track_num]['size'][-1]
        track_sizes[active_num] = track_size
        
    return track_sizes
    

def filter_tracks_without_new_points(track_list, distance, row_ind, 
                                     col_ind, active_list, frame_ind,
                                     debug=False, use_size=True):
    """ Deal with instances where some tracks don't have new points.
    
    This happens when there isn't a new point close enough to existing tracks
    or when there are fewwer new points than existing tracks. When it is the
    later, the longer track takes presedence.
    
    Args:
        track_list (list): all tracks
        distance (np array): distances for every old point new point pair
        row_ind (np array): from linear sum assignment
        col_ind (np array): from linear sum assignment
        active_list (list): list of active track indexes
        frame_ind (int): current frame index
    """
    
    row_ind_full = np.arange(len(active_list))
    col_ind_full = np.zeros(len(active_list), dtype=int)
    duplicates = []
    to_delete = [] # less competive track for the same point, or nothing close
    
    for r_ind in row_ind_full:
        if r_ind in row_ind:
            # This track has been paired to a new point
            col_ind_full[r_ind] = col_ind[np.where(row_ind == r_ind)]
            if debug:
                # add debug info
                track_list[active_list[r_ind]]['debug'].append(
                    '{} assigned normal point'.format(frame_ind))
        else:
            # This track wasn't assigned to a new point with the linear sum assignment
            if np.min(distance[r_ind]) < track_list[active_list[r_ind]]['max_distance']:
                # There is a new point within this tracks assignment range
                duplicates.append(np.argmin(distance[r_ind]))
                col_ind_full[r_ind] = duplicates[-1]
                if debug:
                    # add debug info
                    track_list[active_list[r_ind]]['debug'].append(
                        '{} wasn\'t assigned point but one is near'.format(frame_ind))
            else:
                # This track wasn't assigned a new point and their isn't one close by
                to_delete.append(r_ind)
                if debug:
                    # add debug info
                    track_list[active_list[r_ind]]['debug'].append(
                        '{} wasn\'t assigned point and none near'.format(frame_ind))
                
    track_lengths = _get_track_lengths(frame_ind, track_list, active_list)
    if use_size:
        track_sizes = _get_track_sizes(frame_ind, track_list, active_list)

    for duplicate in duplicates:
        competing_tracks = np.squeeze(np.argwhere(col_ind_full == duplicate))
        longest_track = np.max(track_lengths[competing_tracks])
        if np.sum(track_lengths[competing_tracks]==longest_track) > 1:
            if use_size:
                dominant_track_ind = np.argmax(track_sizes[competing_tracks])
            else:
                # Just takes the first track with max length
                dominant_track_ind = np.argmax(track_lengths[competing_tracks])
        else:
            dominant_track_ind = np.argmax(track_lengths[competing_tracks])
        
        # Tracks that want the same point but are shorter
        # (Following 3 lines remove all but dominant track_ind)
        to_delete.extend(competing_tracks[:dominant_track_ind])
        if dominant_track_ind  < len(competing_tracks):
            to_delete.extend(competing_tracks[dominant_track_ind+1:])
    if to_delete:
        to_delete_a  = np.array(to_delete)
        col_ind_full = np.delete(col_ind_full, to_delete_a)
        row_ind_full = np.delete(row_ind_full, to_delete_a)
        if debug:
            # add debug info
            for ind in to_delete_a:
                track_list[active_list[ind]]['debug'].append(
                        '{} no new point given'.format(frame_ind))
            
    
    return row_ind_full, col_ind_full

def fix_tracks_with_small_points(track_list, distance, row_ind, 
                                     col_ind, active_list, size_list, frame_ind,
                                debug=False):
    """ Big bat tracks should connect to big points.
    
    Sometimes noise pops up next to a bat and is used instead of the next bat
    point. Usually there is an obvious size difference.
    
    Args:
        track_list (list): all tracks
        distance (np array): distances for every old point new point pair
        row_ind (np array): from linear sum assignment
        col_ind (np array): from linear sum assignment
        active_list (list): list of active track indexes
        size_list (list): sizes of all new possible points
        frame_ind (int): current frame index
    """
    
    row_ind_full = np.arange(len(active_list))
    col_ind_full = np.zeros(len(active_list), dtype=int)
    duplicates = []
    to_delete = [] # less competive track for the same point, or nothing close
    
    for r_ind in row_ind_full:
        if r_ind in row_ind:
            # This track has been paired to a new point
            col_ind_full[r_ind] = col_ind[np.where(row_ind == r_ind)]
            track = track_list[active_list[r_ind]]
            # if new point is less than 20% of last point probably something different
            min_new_size = track['size'][-1] / 5
            if min_new_size < 2:
                # old point is already small, don't worry about it
                continue
            new_size = size_list[col_ind[np.where(row_ind == r_ind)]]
            if new_size < min_new_size:
                potential_new_inds = np.argwhere(size_list > min_new_size)
                # take the closest new point that is a reasonable size
                if np.any(potential_new_inds):
                    new_ind = np.argmin(distance[r_ind, potential_new_inds])
                    if distance[r_ind, potential_new_inds[new_ind]] < track['max_distance']:
                        duplicates.append(potential_new_inds[new_ind])
                        col_ind_full[r_ind] = duplicates[-1]
                        if debug:
                            # add debug info
                            track_list[active_list[r_ind]]['debug'].append(
                                '{} assigned too small point'.format(frame_ind))
                
    track_lengths = _get_track_lengths(frame_ind, track_list, active_list)
    track_sizes = _get_track_sizes(frame_ind, track_list, active_list)

    for duplicate in duplicates:
        competing_tracks = np.squeeze(np.argwhere(col_ind_full == duplicate))
        if not competing_tracks.shape:
            # There is only one track after this point
            continue
        longest_track = np.max(track_lengths[competing_tracks])
        if np.sum(track_lengths[competing_tracks]==longest_track) > 1:
            dominant_track_ind = np.argmax(track_sizes[competing_tracks])
        else:
            dominant_track_ind = np.argmax(track_lengths[competing_tracks])
        
        # Tracks that want the same point but are shorter
        to_delete.extend(competing_tracks[:dominant_track_ind])
        if dominant_track_ind  < len(competing_tracks):
            to_delete.extend(competing_tracks[dominant_track_ind+1:])
    if to_delete:
        to_delete_a  = np.array(to_delete)
        col_ind_full = np.delete(col_ind_full, to_delete_a)
        row_ind_full = np.delete(row_ind_full, to_delete_a)
        if debug:
            # add debug info
            for ind in to_delete_a:
                track_list[active_list[ind]]['debug'].append(
                        '{} no new point given after too small'.format(frame_ind))

    
    return row_ind_full, col_ind_full


def create_max_distance_array(distance, track_list, active_list):
    """Create array that contains max acceptable distance between all pairs of points.
    
    Args:
        distance (np array): distance between all new and old points
        track_list (list): list of all tracks
        active_list (list): indexes of all active tracks
        
    return array of same size as distance
    """
    
    max_distance = np.zeros_like(distance)

    for active_num, track_num in enumerate(active_list):
        max_distance[active_num, :] = track_list[track_num]['max_distance']
        
    return max_distance

def filter_bad_assigns(track_list, active_list, distance, max_distance, row_ind, 
                       col_ind, double_assign=False, debug=False):
    """ Deal with instances where point is assigned to track that is too far.
    
    Args:
        track_list (list): list of all tracks
        active_list (list): inds of all currently active tracks
        distance (np array): distance between all pairs of old and new points
        max_distance (np array): max allowed distance between every new and old point
        row_ind (np array): row index for each active track
        col_ind (np array): col_index for reach active track
        double_asign (boolean): all two tracks assigned to same point
    """
    
    bad_assign = distance[row_ind, col_ind] > max_distance[:, 0][row_ind]

    if np.any(bad_assign):
        bad_assign_points = np.where(bad_assign)[0]

        # Assign multiple tracks to nearby points, 
        # in cases where track got assigned to somewhere far away
        # because closer track got assigned to point first
        # this case could come up when two animals get too close 
        # so they merge to one point
        if double_assign:
            col_ind[bad_assign_points] = np.argmin(
                distance[row_ind[bad_assign_points],:], 1)

        # There may be some tracks that just don't have any new points near by.  
        # Filter those out
        not_valid_assign = distance[row_ind, col_ind] > max_distance[:, 0][row_ind]
        if np.any(not_valid_assign):
            if debug:
                for r_ind in np.argwhere(not_valid_assign):
    #                 print('test', np.argwhere(not_valid_assign), r_ind, not_valid_assign)
                    track_list[active_list[r_ind[0]]]['debug'].append('Distance is too far to next point.')
            valid_assign = distance[row_ind, col_ind] <= max_distance[:, 0][row_ind]
            col_ind = col_ind[valid_assign]
            row_ind = row_ind[valid_assign]
    
    return row_ind, col_ind


def process_points_without_tracks(distance, max_distance, track_list, 
                                  new_positions, contours=None, frame_ind=None, 
                                  sizes=None, noise=1):
    """ Find all points that are too far away from existing tracks and create new tracks.
    
    Args:
        distance (np array): distance between every new and old point
        max_disatnce (np array): max allowed distance between every new and old point
        track_list (list): list of all tracks
        new_positions (np array): posible new positions in next frame
        contours (list): list of all contours in frame
        frame_ind (int): current frame number   
        sizes (np array): all sizes of detections in currrent frame
        noise: noise value for new tracks
        """
    
    is_max_distance = distance > max_distance
    # New point is too far away from every existing track
    new_track = np.all(is_max_distance, 0)
    new_track_ind = None
    new_position_indexes = np.arange(new_positions.shape[0])
    if np.any(new_track):
        new_track_ind = np.where(new_track)[0]
        for ind in new_track_ind:
            if contours is not None:
                contour = contours[ind]
            else:
                contour = None
            if sizes is not None:
                size = sizes[ind]
            else:
                size = None
            track_list.append(
                create_new_track(first_frame=frame_ind, 
                                 first_position=new_positions[ind], 
                                 pos_index=ind, noise=noise, 
                                 contour=contour, size=size
                                )
            )
            
        # Get rid of new points that are too far away and were 
        # just added as new tracks
        distance = np.delete(distance, new_track_ind, 1)
        new_positions = np.delete(new_positions, new_track_ind, 0)
        new_position_indexes = np.delete(new_position_indexes, new_track_ind)
        if sizes is not None:
            sizes = np.delete(sizes, new_track_ind)
        if contours is not None:
            for track_ind in new_track_ind[::-1]:
                contours.pop(track_ind)
    if (sizes is not None) and (contours is not None):
            return track_list, distance, new_positions, new_position_indexes, sizes, contours
    elif (sizes is not None) or (contours is not None):
        print("Warning sort out return statement for this case if you want to use.")
        return None
    else:
        return track_list, distance, new_positions, new_position_indexes,

def finalize_track(track):
    """Call when track is finished to turn track lists in into numpy arrays.
    
    When tracks are being created positions and position indexes are added to list
    for efficieny since final size of array is unknown while track is still being
    created.
    
    Args:
        track: track dict as defined in create_new_track
        
    Return track dict with 'track' and 'pos_index' items as arrays
    
    """
    track['track'] = np.stack(track['track'])
    track['pos_index'] = np.expand_dims(np.stack(track['pos_index']), 1)
    if 'size' in track:
        track['size'] = np.stack(track['size'])
    
    return track

def finalize_tracks(track_list):
    """ Convert tracks to array and get rid of extra points at end.
    
    Args:
        track_list (list): list of all tracks
        
    return modified track list"""
    for track_ind, track in enumerate(track_list):
        track = finalize_track(track)
        #number of extra points at the end of track that were added hoping 
        #that the point would reapear nearby.  Since the tracking is now 
        # finished.  We can now get rid of these extra points tacked on to the end
        old_shape = track['track'].shape[0]
        last_real_index = track['last_frame'] - track['first_frame'] + 1
        track['track'] = track['track'][:last_real_index]
        track['pos_index'] = track['pos_index'][:last_real_index]
        if 'size' in track:
            track['size'] = track['size'][:last_real_index]
        if 'contour' in track:
            track['contour'] = track['contour'][:last_real_index]
        track_list[track_ind] = track
    return track_list
        
#returns an array of shape (len(active_list), positions1.shape[0])
#row is distance from every new point to last point in row's active list
def calculate_distances(new_positions, track_list, active_list):
    """ Calculate the distance between every new position and every active track.
    Distance between 
    
    Args:
        new_positions (numpy array): p x 2
        track_list (list): list of all tracks
        active_list (list): index of tracks that could still be added to
        
    return 2d array of distance betwen every combination of points
    
    """
    #positions from last step
    old_positions = [track_list[track_num]['track'][-1] for track_num in active_list]
    old_positions = np.stack(old_positions)

    x_diff = (np.expand_dims(new_positions[:, 1], 0) 
              - np.expand_dims(old_positions[:, 1], 1)
             )

    y_diff = (np.expand_dims(new_positions[:, 0], 0) 
              - np.expand_dims(old_positions[:, 0], 1)
             )
    return np.sqrt(x_diff ** 2 + y_diff ** 2)

def calculate_active_list(track_list, max_unseen_time, frame_num, debug=False):
    active_list = []
    for track_num in range(len(track_list)):
        if frame_num - track_list[track_num]['last_frame'] <= max_unseen_time:
            active_list.append(track_num)
        else:
            if debug:
                track_list[track_num]['debug'].append('{} no longer active'.format(frame_num))
    return active_list

def get_confirmed_inds(track_list):
    """ Return indexes of all tracks that have crossed over the noise threshold."""
    confirm_list = []
    for track_num in range(len(track_list)):
        if track_list[track_num]['noise'] <= 0:
            confirm_list.append(track_num)
    return np.array(confirm_list)

def calculate_max_distance(track_list, active_list, max_distance, 
                           max_distance_noise, min_distance, use_size=False,
                           size_dict=None, min_distance_big=None):
    
    """ Calculate the max distance to search for new points for each track.
    
    The max distance is determined by the minimum of a fixed upper threshold
    or .45 x the distance to the closest neighbor. However, established tracks
    defined by those that have a noise value of 0 or below are considered 
    possible neighbors. This means new tracks don't restrict existing tracks 
    search area. A minimum distance also sets a floor on the seach distance such 
    that very close neighbors don't limit all possible connections. 
    
    Args:
        track_list: list of all tracks
        active_list: list of tracks that could be added to
        max_distance: upper distance threshold
        max_distance_noise: upper distance threshold tracks with noise values
            above 0
        min_distance: lowwer distance threshold
        use_size: if objects have assosiated size and want to use that for 
            additional rules
        size_dict: information about point sizes
        min_distance_big: min distance for large sized points. 
    """
    
    # only check distances to established tracks defined by a noise value of 
    # 0 or below
    positions0 = [track_list[active_list[0]]['track'][-1]]
    if len(active_list) > 1:
        for track_num in active_list[1:]:
            if track_list[track_num]['noise'] <= 0:
                positions0.append(track_list[track_num]['track'][-1])
    positions0 = np.stack(positions0)
                
    distance = calculate_distances(positions0, track_list, active_list)
    # closest point will be itself, so make zero distance 
    # bigger than other distances
    distance[np.where(distance == 0)] = float("inf")
    # HYPER PARAMETER
    # Don't connect to points that are closer to other points
    closest_neighbor = np.min(distance, 1) * .45
    # Even if neighbors are all far away, have a max threshold to look for new points 
    closest_neighbor[np.where(closest_neighbor > max_distance)] = max_distance
    # However, even if neighbors are very close, should be able to connect
    # to points within radius of min distance
    closest_neighbor[np.where(closest_neighbor < min_distance)] = min_distance
    for active_ind, track_num in enumerate(active_list):
        track_list[track_num]['max_distance'] = closest_neighbor[active_ind] 
        if track_list[track_num]['noise'] > 0:
            if closest_neighbor[active_ind] > max_distance_noise:
                track_list[track_num]['max_distance'] = max_distance_noise
                
        if use_size:
            # Only is objects have related size (added for bats)
            size = track_list[track_num]['size'][-1]
            max_distance = track_list[track_num]['max_distance']
            if size < 30:
                max_distance = np.min([15, max_distance]) 
            elif size < 120:
                max_distance = np.min([20, max_distance])
            elif not np.isnan(size):
                if min_distance_big:
                    # Even is points near by, give room to look around
                    max_distance = np.max([min_distance_big, max_distance])
            
            track_list[track_num]['max_distance'] = max_distance
    return track_list

def add_interpolated_points(frame_ind, track, new_position):
    """ Replace points since last seen with interpolated estimates.
    
    Args:
        frame_ind (int): current frame index in observation
        track (track obj): the track that is being modified
        new_position (np array): new point being added
        
    return updated track
    """
    
    gap_distance = (new_position - track['track'][-1])
    
    missed_steps = frame_ind - track['last_frame'] - 1
    step_distance = gap_distance / (missed_steps + 1)
    for step in range(missed_steps):
        track['track'][-step - 1] = (track['track'][-step - 1] 
                                     + (missed_steps - step) * step_distance)
        
    return track

def remove_noisy_tracks(track_list, max_noise=2):
    """ Delete tracks that have noise values that are too high.
    
    Args:
        track_list (list): list of all tracks
        max_noise: remove track if track has this noise value or higher
        
    return modified track list
    """
    
    # Traverse the list in reverse order so if there are multiple tracks that
    # need to be removed the indexing doesn't get messed up 
    for track_num in range(len(track_list) - 1, -1, -1): 
        if track_list[track_num]['noise'] >= max_noise:
            del track_list[track_num]

    return track_list

def create_tracks_for_leftover_points(track_list, col_inds, frame_ind, 
                                      min_new_track_distance, distance, 
                                      new_positions, new_position_indexes,
                                      contours=None, sizes=None, noise=1):
    
    """ There can be new points that are close enough to existsing
    tracks to prevent them from being added in the beginning that don't
    end up being connected to existing tracks. This are added now.
    
    Args:
        track_list (list): list of all tracks
        col_inds (np array): assossiates new points to columns in distance
        frame_ind (int): current frame number in observation
        min_new_track_distance (int): closest new point can be to existing tracks
        distance (np array): distance between all existing tracks and new points
        new_positions (np.array): location of all new points n x 2
        new_position_indexes (np array): raw index of each new point
        contours (list): list of all contours
        sizes (list): all sizes in frame
        noise: noise value for new tracks
    """
    

    # There are possible new points
    for pos_ind in range(new_positions.shape[0]):
        if pos_ind in col_inds:
            # This point was already added to an existing track
            continue
        # Only add points that aren't too close to existing tracks 
        # This just a conservative choice in case of an object 
        # being detected twice and creating two tracks that cause trouble
        # for each other
        if np.min(distance[:, pos_ind]) > min_new_track_distance:
            # This new point isn't too close to existing tracks
            if contours is not None:
                contour = contours[pos_ind]
            else:
                contour = None
            if sizes is not None:
                size = sizes[pos_ind]
            else:
                size = None
            track_list.append(
                create_new_track(first_frame=frame_ind, 
                                 first_position=new_positions[pos_ind], 
                                 pos_index=new_position_indexes[pos_ind], 
                                 noise=noise, 
                                 contour=contour,
                                 size=size
                                )
            )
    return track_list
    
    
    
def update_tracks(track_list, active_list, frame_index, row_inds, col_inds,
                  new_positions, new_position_indexes, new_sizes=None,
                  new_contours=None, distance=None, min_new_track_distance=None, 
                  debug=False, new_track_noise=1):
    """ Update tracks depending on if they have new points or not.
    
    Args:
        track_list (list): list of all tracks
        active_list (list): list of indexes of active tracks
        frame_index (int): current frame index
        row_inds (np array): links active tracks to distance array
        col_inds (np array): links new points to distance array
        new_positions (np array): positions of new points n x 2
        new_position_indexes (np array): raw indexes of new points
        new_sizes (list): sizes of new points
        new_contours (list): list of new contours
        distance (np array): distance between all tracks and new points
        min_new_track_distance (int): how close are new points allowed
            to be to old points to start new track
        
    return updated track list
        """
    
    active_list = np.array(active_list)
    for track_num, track in enumerate(track_list):
        if track_num in active_list:
            if row_inds is None:
                track_list[track_num] = update_unmatched_track(track)
                continue
            if track_num in active_list[row_inds] and len(new_positions) != 0:
                row_count = np.where(track_num == active_list[row_inds])[0]
                new_position = new_positions[col_inds[row_count[0]]]
                if track['last_frame'] != frame_index - 1:
                    # This is a refound track, linearly interpolate 
                    # from when last seen
                    track = add_interpolated_points(frame_index, track, new_position)
                new_position_index = new_position_indexes[col_inds[row_count[0]]]
                if new_sizes is not None:
                    new_size = new_sizes[col_inds[row_count][0]]
                else:
                    new_size = None
                if new_contours is not None:
                    new_contour = new_contours[col_inds[row_count][0]]
                else:
                    new_contour = None
                track_list[track_num] = update_matched_track(
                    track, frame_index, new_position, new_position_index,
                    size=new_size, contour=new_contour
                )
                if debug:
                    track_list[track_num]['debug'].append(
                        '{} row_ind: {}, col_ind: {} pos_ind: {} dist: {}'.format(
                            frame_index, row_inds[row_count[0]], col_inds[row_count[0]], 
                            new_position_index, distance[row_inds[row_count[0]], 
                                                         col_inds[row_count[0]]]
                        )
                    )
            else:
                track_list[track_num] = update_unmatched_track(track)
    
    
    if distance is not None:
        if distance.shape[0] < distance.shape[1]:
            # Add new tracks for new points that weren't added to existing tracks 
            # but weren't far enough away before to aleady get a new track
            track_list = create_tracks_for_leftover_points(
                track_list, col_inds, frame_index, min_new_track_distance, 
                distance, new_positions, new_position_indexes, new_contours,
                new_sizes, noise=new_track_noise
            )

    return track_list

def find_tracks(first_frame_ind, positions, params, 
                contours_files=None, contours_list=None,
                sizes_list=None, max_frame=None, verbose=True, 
                tracks_file=None, detection_dicts=None):
    """ Take in positions of all individuals in frames and find tracks.
    
    Args: 
        first_frame_ind (int): index of first frame of these tracks
        positions (list): n x 2 for each frame
        params: dict containing:
            max_distance_threshold: max distance from existing tracks that
                new points can be connected to
            max_distance_threshold_noise: max distance from existing tracks that
                new points can be connected to for tracks that have positive
                noise values
            min_distance_threshold: minimum max distance threshold for tracks
                even if there are other tracks that are even closer
            max_unseen_time: number of frames a track can remain active without
                new points being added to it
            min_new_track_distance: only points above this distance to 
                existing tracks will be used to start new tracks
            new_track_noise: noise value to give new tracks. This value
                will decrease by one each time a point is added to the track
            max_noise_value: tracks with this noise value or above will be 
                removed
        contours_files (list): list of files for contour info from each frame
        contours_list: already loaded list of contours, only used if contours_file
            is None
        sizes_list (list): sizes info from each frame
        detection_dicts: list of dictionaries the store the model output for each
            frame. Can contain keys like, 'bboxes', 'scores', 'pred_classes'
    
    return list of all tracks found
    """
    
    raw_track_list = []

    max_distance_threshold = params['max_distance_threshold']
    max_distance_threshold_noise = params['max_distance_threshold_noise']
    min_distance_threshold = params['min_distance_threshold'] 
    max_unseen_time = params['max_unseen_time'] 
    min_new_track_distance = params['min_new_track_distance'] 
    new_track_noise = params['new_track_noise'] 
    max_noise_value = params['max_noise_value'] 


    if max_frame is None:
        max_frame = len(positions)
    
    for frame_ind in range(first_frame_ind, max_frame):
        
        # get tracks that are still active (have been seen within the specified time)
        active_list = calculate_active_list(raw_track_list, max_unseen_time, frame_ind)
        
        if verbose:
            if frame_ind % 10000 == 0:
                print('frame {} processed.'.format(frame_ind))
                if tracks_file:
                    np.save(tracks_file, np.array(raw_track_list, dtype=object))
        if len(active_list) == 0:
            #No existing tracks to connect to
            #Every point in next frame must start a new track
            raw_track_list = add_all_points_as_new_tracks(
                raw_track_list, frame_ind, positions[frame_ind], 
                class_labels=detections_dicts[frame_ind]['pred_classes'],
                noise=new_track_noise
            )
            continue

        
        new_positions = None
        row_ind = None
        col_ind = None
        new_sizes = None
        new_position_indexes = None
        distance = None
        contours = None
        # Check if there are new points to add
        if len(positions[frame_ind]) != 0:
            
            #positions from the next step
            new_positions = positions[frame_ind]
            
            # Calculate upper threshold for connecting new points for each track
            raw_track_list = calculate_max_distance(
                raw_track_list, active_list, max_distance_threshold, 
                max_distance_threshold_noise, min_distance_threshold
            )
            # Calculate distance to each new point
            distance = calculate_distances(
                new_positions, raw_track_list, active_list
            )
            # Creates array of same size as 'distance' where each row contains
            # the max distance threshold for the coresponding existing track
            max_distance = create_max_distance_array(
                distance, raw_track_list, active_list
            )
            
            # Some new points could be too far away from every existing track
            raw_track_list, distance, new_positions, new_position_indexes = process_points_without_tracks(
                distance, max_distance, raw_track_list, new_positions, 
                frame_ind=frame_ind, noise=new_track_noise
            )
            
                
            if distance.shape[1] > 0:
                # There are new points can be assigned to existing tracks
                # connect the dots from one frame to the next
                # Note: using log distance so that the effect of long pairs
                # contributes less to overal optimzation compared to close pairs.
                # One is added to distance so the log is lower bounded by 0
                row_ind, col_ind = linear_sum_assignment(np.log(distance + 1))
                        
                # In casese where there are fewer new points than existing tracks
                # some tracks won't get new point. Check if those tracks have 
                # points within distance threshold. If so, if those tracks are 
                # longer than the track that was assigned point, switch that 
                # point to the longer track.
                row_ind, col_ind = filter_tracks_without_new_points(
                    raw_track_list, distance, row_ind, col_ind, active_list, 
                    frame_ind, use_size=False
                )
                # see if points got assigned to tracks that are farther 
                # than max_threshold_distance, This happens when all valid possible 
                # points ended up being assigned to other points.
                row_ind, col_ind = filter_bad_assigns(
                    raw_track_list, active_list, distance, max_distance, row_ind, 
                    col_ind, double_assign=False
                )


        raw_track_list = update_tracks(raw_track_list, active_list, frame_ind, 
                                       row_ind, col_ind, new_positions, 
                                       new_position_indexes, 
                                       distance=distance, 
                                       min_new_track_distance=min_new_track_distance, 
                                       new_track_noise=new_track_noise)
        raw_track_list = remove_noisy_tracks(
            raw_track_list, max_noise=max_noise_value
        )
    raw_track_list = finalize_tracks(raw_track_list) 
    if tracks_file:
        np.save(tracks_file, np.array(raw_track_list, dtype=object))
        print('{} final save.'.format(os.path.basename(os.path.dirname(tracks_file)))) 
    return raw_track_list