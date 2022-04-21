import cv2
import numpy as np
import glob
import copy
import os
import pandas as pd

def sort_by_last_frame(item):
    return item['last_frame']

def sort_by_first_frame(item):
    return item['first_frame']

class TrackCorrectionGUI():
    
    def __init__(self, tracks_path, frame_files, factor,
                 positions_path="", point_scale=1.0, show_all_tracks=True,
                 xy_tracks=False):
        """ show_all_tracks: whether to show pieces of all other active tracks in frame."""
        self.positions_path = positions_path
        self.tracks_path = tracks_path
        self.files = frame_files
        self.show_all_tracks = show_all_tracks
        self.xy_tracks = xy_tracks
        if os.path.isfile(positions_path):
            print('positions loaded')
            self.listofpositions = np.load(positions_path, allow_pickle=True)
        else:
            self.listofpositions = None

        # work on a copy of the tracks info.  Don't edit the original 
        self.listoftracks = np.ndarray.tolist(copy.deepcopy(np.load(tracks_path, allow_pickle=True)))
        # Focal tracks will only be sorted once
        # Use case is start from first frame and build the track to the end
        self.focal_tracks = copy.copy(self.listoftracks)
        self.focal_tracks.sort(key=sort_by_first_frame)
        self.listoftracks.sort(key=sort_by_first_frame)
        # only do something on key release
        # so only act when key press was something but now released
        self.key_press = None
            

        for track in self.listoftracks:
            track['connected'] = []
            track['remove'] = False
            if not 'pos_index' in track.keys():
                pos_index = np.array([np.nan for _ in track['track']])
                pos_index = np.expand_dims(pos_index, 1)
                track['pos_index'] = pos_index

        self.factor = factor
        self.point_scale = point_scale
        image = cv2.imread(self.files[0]) #to get size
        self.h = int(np.size(image,0)*factor)
        self.w = int(np.size(image,1)*factor)
        self.full_pic = np.zeros((int(self.h),self.w,3), dtype=np.uint8)
        self.focaltrackcount = 0
        self.trackcount = 1
        self.framecount = 0
        temp_frame = self.focal_tracks[0]['last_frame'] #to get good frame
        if temp_frame >= len(self.files):
            self.framecount = int(len(self.files)) - 1
        else:
            self.framecount = temp_frame
            
        self.num_corrections = 0
        
        self.tracks_stack = []
        
        self.hide = False
    
    # call after every track change and will deal with saving at the correct user specified times
    def save(self, save_type='passive'):
        # How many save files to create
        save_n_copies = 2
        # How often to save after making corrections
        save_every_n_corrections = 10
        if self.num_corrections % save_every_n_corrections == 0 or save_type=='active':
            if save_type == 'passive':
                file_name = (os.path.splitext(self.tracks_path)[0] + '-' +
                             str(self.num_corrections / save_every_n_corrections % save_n_copies) + '.npy')
            else:
                file_name = (os.path.splitext(self.tracks_path)[0] + '-' +
                             'final.npy')
            np.save(file_name, self.listoftracks)

            print('saved at', file_name)
        self.num_corrections += 1
    
    #draw points on image
    def draw_points(self, picture, listofpositions, color1,color2,color3, r):
        for i in listofpositions:
            if np.isnan(i[1]):
                continue
            r = int(r)
            r = np.max([r, 1])
            if self.xy_tracks:
                cv2.circle(picture, (int(i[0]), int(i[1])), r, (color1,color2,color3), -1)
            else:
                cv2.circle(picture, (int(i[1]), np.size(picture,0) - int(i[0])), r, (color1,color2,color3), -1)

    #to process clicking on a button
    def clicked(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Create new track at point where user clicks if they press shift key while clicking
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.update_stack()
                if self.xy_tracks:
                    x = int((1/self.factor) * x) #to adjust for picture
                    y = int((1/self.factor) * y) #to adjust for picture
                    track = np.array([[x, y]])
                else:
                    x = int((1/self.factor) * x) #to adjust for picture
                    y = int((1/self.factor) * self.h - (1/self.factor) * y) #to adjust for picture
                    track = np.array([[y, x]])
                pos_index = np.array([[np.nan]])
                new_track_dict = self._create_new_track(self.framecount, track, pos_index, class_label=None)
                first_frame_array =  np.array([track['first_frame'] for track in self.focal_tracks])
                new_track_ind = np.searchsorted(first_frame_array, self.framecount)
                self.focal_tracks.insert(new_track_ind, new_track_dict)
                self.listoftracks.append(new_track_dict)
                self.listoftracks.sort(key=sort_by_first_frame)
                self.focaltrackcount = new_track_ind

                self.save()
            elif flags & cv2.EVENT_FLAG_CTRLKEY:
                self.connect_from_out_of_frame(x,y)
            
            else:
                self.add_point(x,y)

    def draw_window(self):
        image = cv2.imread(self.files[self.framecount]) #read image
        point_size = 2 * self.point_scale
        text_size = 1
        track_length = 40

        #draw positions in the current frame
        image = cv2.resize(image, (self.w, self.h))
        
        
        if not self.hide:
            if self.listofpositions is not None:
                positions = self.listofpositions[self.framecount] 
                self.draw_points(image, positions*self.factor, 0,0,255, int(point_size*3))

            # show segments of all active tracks in the current frame
            if self.show_all_tracks:
                for track in self.listoftracks:
                    if track['first_frame'] < self.framecount and track['first_frame'] + len(track['track']) > self.framecount:
                        relative_frame = self.framecount - track['first_frame']
                        if relative_frame - track_length // 2 < 0:
                            track_show = track['track'][:relative_frame+int(track_length/2)]
                        else:
                            track_show = track['track'][relative_frame-int(track_length/2):relative_frame+int(track_length/2)]
                        self.draw_points(image, track_show*self.factor, 0,0,0, int(point_size / 2)) # draw focal track

            # draw circle around focal track
            point = self.focal_tracks[self.focaltrackcount]['track'][-1,:]
            if self.xy_tracks:
                cv2.circle(image, 
                           (int(point[0] * self.factor), int(point[1] * self.factor)), 
                           50, (50, 0, 200), 1
                          )
            else:
                cv2.circle(image, 
                           (int(point[1] * self.factor), int(image.shape[0] - point[0]*self.factor)), 
                           50, (50, 0, 200), 1)

            rel_framecount = self.framecount - self.focal_tracks[self.focaltrackcount]['first_frame']
            if rel_framecount >= 0:
                # Drawing focal track
                self.draw_points(image, 
                                 self.focal_tracks[self.focaltrackcount]['track'][:rel_framecount]*self.factor, 
                                 20,255,255, int(point_size/2)
                                ) 
                self.draw_points(image, 
                                 self.focal_tracks[self.focaltrackcount]['track'][rel_framecount:]*self.factor, 
                                 255,255,200, int(point_size/2)
                                ) 
                self.draw_points(image, 
                                 self.focal_tracks[self.focaltrackcount]['track'][rel_framecount:rel_framecount+1]*self.factor, 
                                 102, 0, 51, point_size*2
                                )
            # Drawing possible new track
            self.draw_points(image, 
                             self.listoftracks[self.trackcount]['track']*self.factor, 
                             255,0,0, int(point_size/2)
                            ) 
            #show where the new track starts
            self.draw_points(image, 
                             self.listoftracks[self.trackcount]['track'][:1]*self.factor, 
                             255,144,30, point_size*2
                            ) 
            rel_frame_new_track = self.framecount - self.listoftracks[self.trackcount]['first_frame']
            # The potential new track starts before the current frame
            if rel_frame_new_track > 0:
                # The potential new track ends after the current frame
                if self.listoftracks[self.trackcount]['first_frame'] + len(self.listoftracks[self.trackcount]['track']) > self.framecount:
                    current_point = self.listoftracks[self.trackcount]['track'][rel_frame_new_track]
                    self.draw_points(image, [current_point*self.factor], 181,186,10, point_size*2)

        
                    
                
#         new_pic = cv2.resize(image, (self.w, self.h))
        self.full_pic[0:self.h,0:self.w] = image[0:self.h,0:self.w] #put onto frame
        frame_diff = self.listoftracks[self.trackcount]['first_frame'] - self.framecount
        font_color = (255, 255, 255)
        
        text_spacing = 400
        text_row = 40
        cv2.putText(self.full_pic, 'frames ahead: %d'%frame_diff, (self.w-text_spacing, text_row), 
                    cv2.FONT_HERSHEY_DUPLEX, text_size, font_color, 1, cv2.LINE_AA)
        cv2.putText(self.full_pic, 'current frame: %d'%self.framecount, (self.w-text_spacing*2, text_row),
                    cv2.FONT_HERSHEY_DUPLEX, text_size, font_color, 1, cv2.LINE_AA)
        cv2.putText(self.full_pic, 'current track: %d'%self.focaltrackcount, (self.w-text_spacing*3, text_row),
                    cv2.FONT_HERSHEY_DUPLEX, text_size, font_color, 1, cv2.LINE_AA)
        try:
        
            diff_between_tracks = (self.listoftracks[self.trackcount]['track'][0, :] - 
                                   self.focal_tracks[self.focaltrackcount]['track'][-1, :])
            dist_to_next = np.sqrt(np.sum((diff_between_tracks) ** 2))
        
            cv2.putText(self.full_pic, 'dist: %d'%dist_to_next, (self.w-text_spacing*4, text_row),
                        cv2.FONT_HERSHEY_DUPLEX, text_size, font_color, 1, cv2.LINE_AA)
            cv2.putText(self.full_pic, 'TFF: %d'%len(self.listoftracks), (self.w-text_spacing*5, text_row),
                        cv2.FONT_HERSHEY_DUPLEX, text_size, font_color, 1, cv2.LINE_AA)
            cv2.putText(self.full_pic, 'focal first frame: %d'%self.focal_tracks[self.focaltrackcount]['first_frame'], 
                        (self.w-text_spacing * 6, text_row),
                        cv2.FONT_HERSHEY_DUPLEX, text_size, font_color, 1, cv2.LINE_AA)
            cv2.putText(self.full_pic, 'focal length: %d'%len(self.focal_tracks[self.focaltrackcount]['track']), 
                        (self.w-text_spacing * 7, text_row),
                        cv2.FONT_HERSHEY_DUPLEX, text_size, font_color, 1, cv2.LINE_AA)
            cv2.putText(self.full_pic, 'add length: %d'%len(self.listoftracks[self.trackcount]['track']), 
                        (self.w-text_spacing * 8, text_row),
                        cv2.FONT_HERSHEY_DUPLEX, text_size, font_color, 1, cv2.LINE_AA)
        except Exception as e:
            print(self.listoftracks[self.trackcount]['track'].shape)
            print('trackcount', self.trackcount)
            print('no text')
            print(e)
        cv2.imshow('pic0', self.full_pic)
        
    # Stack of previous states so action can be undone 
    def update_stack(self):
        self.tracks_stack.append(
            [copy.deepcopy(self.listoftracks), copy.deepcopy(self.focal_tracks), copy.deepcopy(self.listofpositions)])
        # 10 is number of previous states to store
        if len(self.tracks_stack) > 10:
            del self.tracks_stack[0]
            
    def update_listoftracks(self):
        self.listoftracks = copy.copy(self.focal_tracks)
        self.listoftracks.sort(key=sort_by_first_frame)
    
    def update_focal_tracks(self):
        self.focal_tracks = copy.copy(self.listoftracks)
        self.focal_tracks.sort(key=sort_by_first_frame)
        
    def record_new_point(self, point, frame):
        self.listofpositions[frame] = np.vstack([self.listofpositions[frame], point])
            
    # add a point to picture and to current track
    def add_point(self, x, y):
        focal_track = self.focal_tracks[self.focaltrackcount]
        self.update_stack()
        frame_dif = self.framecount - focal_track['last_frame']
        
        x = int((1/self.factor) * x) #to adjust for picture
        if self.xy_tracks:
            y = int((1/self.factor) * y) #to adjust for picture
            x_diff = x - focal_track['track'][-1, 0]
            y_diff = y - focal_track['track'][-1, 1]
            if frame_dif != 0:
                position_dif_step = [x_diff / frame_dif, y_diff / frame_dif]
        else:
            y = int((1/self.factor) * self.h - (1/self.factor) * y) #to adjust for picture
            x_diff = x - focal_track['track'][-1, 1]
            y_diff = y - focal_track['track'][-1, 0]
            if frame_dif != 0:
                position_dif_step = [y_diff / frame_dif, x_diff / frame_dif]
        for frame in range(frame_dif):
            focal_track['track'] = np.vstack([focal_track['track'], focal_track['track'][-1,:] + position_dif_step])
            nan = np.empty((1,1))
            nan[:] = np.nan
            focal_track['pos_index'] = np.vstack([focal_track['pos_index'], nan])
            focal_track['last_frame'] += 1
            
        self.focal_tracks[self.focaltrackcount]['connected'].append(len(self.focal_tracks[self.focaltrackcount]['track']))
        self.update_listoftracks()
        self.find_next_track()
        self.draw_window()
        self.save()
        print('adding point')
        
        # add a point to picture and to current track
    def connect_from_out_of_frame(self, x, y):
        focal_track = self.focal_tracks[self.focaltrackcount]
        self.update_stack()
        if self.xy_tracks:
            x = int((1/self.factor) * x) #to adjust for picture
            y = int((1/self.factor) * y) #to adjust for picture
        else:
            x = int((1/self.factor) * x) #to adjust for picture
            y = int((1/self.factor) * self.h - (1/self.factor) * y) #to adjust for picture

        frame_dif = self.framecount - focal_track['last_frame']
        
        if frame_dif > 0:
        
            for frame in range(frame_dif - 1):
                focal_track['track'] = np.vstack([focal_track['track'], np.array([[np.nan, np.nan]])])
                nan = np.empty((1,1))
                nan[:] = np.nan
                focal_track['pos_index'] = np.vstack([focal_track['pos_index'], nan])
                focal_track['last_frame'] += 1
            
            if self.xy_tracks:
                focal_track['track'] = np.vstack([focal_track['track'], 
                                                  np.array([[x, y]])]
                                                )
            else:
                focal_track['track'] = np.vstack([focal_track['track'], 
                                                  np.array([[y, x]])]
                                                )
            nan = np.empty((1,1))
            nan[:] = np.nan
            focal_track['pos_index'] = np.vstack([focal_track['pos_index'], nan])
            focal_track['last_frame'] += 1

            self.focal_tracks[self.focaltrackcount]['connected'].append(
                len(self.focal_tracks[self.focaltrackcount]['track'])
            )
            self.update_listoftracks()
            self.find_next_track()
            self.draw_window()
            self.save()
            print('adding point from out of frame')
        
        

    #find a good track to go to, given the focal track
    def find_next_track(self):
        last_seen = self.focal_tracks[self.focaltrackcount]['last_frame']
        for track_ind, track in enumerate(self.listoftracks): #make trackcount above 0
            frame_diff = track['first_frame'] - last_seen
            if frame_diff >= 0:
                self.trackcount = track_ind 
                break
        self.trackcount = track_ind

    #to help have keyboard shortcuts
    def detect_keys(self, key):

        if key == ord(';'):
            self.change_frame_function(3)
        elif key == ord('k'):
            self.change_frame_function(-3)
        elif key == ord(']'):
            self.change_frame_function(1)
        elif key == ord('['):
            self.change_frame_function(-1)
        elif key == ord('9'):
            self.change_frame_function(-90)
        elif key == ord('0'):
            self.change_frame_function(90)
        elif key == ord('7'):
            self.change_frame_function(-300)
        elif key == ord('8'):
            self.change_frame_function(300)
        elif key == ord('1'):
            self.change_frame_function(-5000)
        elif key == ord('2'):
            self.change_frame_function(5000)
        elif key == 58:
            self.change_frame_function(3)
        elif key == ord('x'):
            self.change_frame_function(30)
        elif key == ord('z'):
            self.change_frame_function(-30)
        # only do the following thing on the key release
        elif key == 255 and self.key_press:
            key = self.key_press
            self.key_press = None
#             print('here ', key, ' : ', self.key_press)
#             if key == 0 or key == 82: #up key = move track forward
            if key == ord('o'):
                self.next_track_function(1)
#             elif key == 1 or key == 84: #down key = move track back
            elif key == ord('l'):
                self.next_track_function(-1)
            elif key == ord('.'): # . key = move focal track forward
                self.next_focal_track_function(1)
            elif key == ord(','): # , key = move focal track back
                self.next_focal_track_function(-1)
#             elif key == 127: #delete key = delete point
#                 self.remove_point_function()
            elif key == 32: #space key = add to track
                self.add_to_track_function()
            elif key == 45: # - key = remove track
                self.delete_focal_track()
            elif key == ord('g'): # 'g' key = green function
                self.green_function()
            elif key == ord('b'): # 'b' key = blue function
                self.blue_function()
            elif key == ord('s'): # 's' key = split focal path
                self.split_track('focal')
            elif key == ord('a'):
                self.split_track('added')
            elif key == ord('u'): # undo function
                self.undo()
            elif key == ord('h'): # hide tracks so easier to see individuals
                self.hide = True
                self.draw_window()
                self.hide = False
            
            
                
        else:
            self.key_press = key

    #functions for buttons
    def change_frame_function(self, increment):
        n = self.framecount + increment
        if n < len(self.files) and n >= 0:
            self.framecount = n
        self.draw_window()
        
    def next_track_function(self, direction):

        n = self.trackcount + direction
        if n < len(self.listoftracks) and n >= 0:
            self.trackcount = n
        self.draw_window()

        
    def undo(self):
        if self.tracks_stack:
            old_tracks = self.tracks_stack.pop()
            self.listoftracks = old_tracks[0]
            self.focal_tracks = old_tracks[1]
            self.listofpositions = old_tracks[2]
            self.draw_window()
            print('undoing last action...')
        else:
            print('Can not undo any more')

    def next_focal_track_function(self, direction):
        #check that doesn't go out of bounds
        n1 = self.focaltrackcount + direction
        if n1 < len(self.focal_tracks) and n1 >= 0:
            self.focaltrackcount = n1
        #to next get good next potential track
        n = self.focal_tracks[self.focaltrackcount]['last_frame'] 
        if n >= len(self.files):
            self.framecount = int(len(self.files)) - 1
        else:
            self.framecount = n
            

        # Make sure there is at least one point in the track
        if self.focal_tracks[self.focaltrackcount]['track'].shape[0] == 0:
            print('Deleting track because empty.')
            self.delete_focal_track()

        self.find_next_track() #make blue track start after green track ends
        self.draw_window()
        
    
    def delete_focal_track(self):
        self.update_stack()
#         self.focal_tracks[self.focaltrackcount]['remove'] = True
        del self.focal_tracks[self.focaltrackcount]
        if self.focaltrackcount >= len(self.focal_tracks):
            self.focaltrackcount = len(self.focal_tracks) - 1 
        # This is to guard against the next track being empty, maybe better to deal with this later on
        else:
            self.focaltrackcount -= 1
        self.update_listoftracks()
        self.find_next_track()
        self.draw_window()
        self.save()
        
        
        
    def add_to_track_function(self):
        try:
            self.update_stack()
            if self.focal_tracks[self.focaltrackcount]['first_frame'] <= self.listoftracks[self.trackcount]['first_frame']:
                focal_track = self.focal_tracks[self.focaltrackcount]
                added_track = self.listoftracks[self.trackcount]
                focal_list = self.focal_tracks
                added_list = self.listoftracks
                focal_index = self.focaltrackcount
                added_index = self.trackcount
            else:
                added_track = self.focal_tracks[self.focaltrackcount]
                focal_track = self.listoftracks[self.trackcount]
                added_list = self.focal_list
                focal_list = self.listoftracks
                added_index = self.focaltrackcount
                focal_index = self.trackcount
                print('flipped')
            if focal_track is added_track: #make sure isn't same as focal track
                pass
            else:
                # record where new points are added
                focal_track['connected'].append(focal_track['track'].shape[0]) 
                # it is possible that the second track starts before the first one ends
                # default behavior:
                # use the part from the added track
                # linear interpolation when there is a gap between tracks
                # otherwise just join
                # overlap = (focal_track['first_frame'] + focal_track['track'].shape[0]) - added_track['first_frame'] # should: focal_track['first_frame'] + focal_track['track'].shape[0] = 'last_frame
                overlap = focal_track['last_frame'] - added_track['first_frame'] + 1
                if overlap > 0:
                    print('1')
                    # added track is completely overlapping with focal track
                    if focal_track['last_frame'] >= added_track['last_frame']:
                        first_frame = added_track['first_frame'] - focal_track['first_frame']
                        last_frame = added_track['last_frame'] - focal_track['first_frame'] + 1
                        focal_track['track'][first_frame:last_frame] = added_track['track']
                        focal_track['pos_index'][first_frame:last_frame] = added_track['pos_index']
                        print('2')
                    # added track extends beyond focal track
                    else:
                        print('overlap' , overlap)
                        print('focal last', focal_track['last_frame'], 'added last', added_track['last_frame'])
                        print('focal first', focal_track['first_frame'], 'added first', added_track['first_frame'])
                        print(focal_track['track'].shape)
                        focal_track['track'] = np.vstack([focal_track['track'][:-overlap], added_track['track']])
                        focal_track['pos_index'] = np.vstack([focal_track['pos_index'][:-overlap], added_track['pos_index']])
                        focal_track['last_frame'] = added_track['last_frame']

                        print(focal_track['track'].shape)
                        
                # there is a gap betwen tracks
                elif overlap <= -1:
                    print('4')
                    position_dif = added_track['track'][0, :] - focal_track['track'][-1, :]
                    # +1 because one dot needs two steps
                    position_dif_step = position_dif / (-1 * overlap + 1)
                    for step in range(-1 * (overlap)):
                        focal_track['track'] = np.vstack([focal_track['track'], focal_track['track'][-1,:] + position_dif_step])
                        nan = np.empty((1,1))
                        nan[:] = np.nan
                        focal_track['pos_index'] = np.vstack([focal_track['pos_index'], nan])
                    focal_track['track'] = np.vstack([focal_track['track'], added_track['track']])
                    focal_track['pos_index'] = np.vstack([focal_track['pos_index'], added_track['pos_index']])
                    focal_track['last_frame'] = added_track['last_frame']
                # perfect alignment
                else:
                    print('5')
                    focal_track['track'] = np.vstack([focal_track['track'], added_track['track']])
                    focal_track['pos_index'] = np.vstack([focal_track['pos_index'], added_track['pos_index']])
                    focal_track['last_frame'] = added_track['last_frame']
                assert (focal_track['first_frame'] + len(focal_track['track']) - 1 == focal_track['last_frame']), " doesn't fit with last_frame value"
                added_track['remove'] = True
                for track_ind, track in enumerate(focal_list):
                    if track['remove']:
                        print('track merged')
                        del focal_list[track_ind]
                self.update_listoftracks()
            self.find_next_track() #make blue track start after green track ends
            self.draw_window()
            self.save()
        except Exception as e:
            print(e)
        
    
    
    def _create_new_track(self, first_frame, track, pos_index, class_label=None):
        new_track = {'track': track,
                    'first_frame': first_frame,
                    'last_frame': first_frame + track.shape[0] - 1, 
                     'connected': [],
                     'pos_index': pos_index,
                     'remove': False,
                     'class_label': [class_label]
                    }

        return new_track
    
        
    def split_track(self, track_type):
        # assumes the user is on the last frame of the old track
        
        self.update_stack()
        if track_type == 'focal':
            track = self.focal_tracks[self.focaltrackcount]
            track_ind = self.focaltrackcount
        elif track_type == 'added':
            track = self.listoftracks[self.trackcount]
            track_ind = self.trackcount
        else:
            raise NameError('not valid track type to print')
            
        if track['last_frame'] == self.framecount:
            return
        
        # plus one because new track starts one after current frame 
        split_frame = self.framecount - track['first_frame'] + 1
        new_track = copy.copy(track['track'][split_frame:])
        new_pos_index = copy.copy(track['pos_index'][split_frame:])
        # Older version had some tracks not have a class
        try:
            new_track_dict = self._create_new_track(self.framecount + 1, new_track, new_pos_index, track['class'])
        except:
            new_track_dict = self._create_new_track(self.framecount + 1, new_track, new_pos_index)

        # get rid of the new track part from the old track
        track['track'] = track['track'][:split_frame]
        track['pos_index'] = track['pos_index'][:split_frame]
        track['last_frame'] = self.framecount
        if track_type == 'focal':
            self.focal_tracks.append(new_track_dict)
            self.focal_tracks.sort(key=sort_by_first_frame)
            self.listoftracks = copy.copy(self.focal_tracks)
        elif track_type == 'added':
            self.listoftracks.append(new_track_dict)
            self.listoftracks.sort(key=sort_by_first_frame)
            self.focal_tracks = copy.copy(self.listoftracks)

        self.save()
        
        

    def blue_function(self):
        n = self.listoftracks[self.trackcount]['first_frame'] #to get good frame
        if n >= len(self.files):
            self.framecount = int(len(self.files)) - 1
        else:
            self.framecount = n
        self.draw_window()

    def green_function(self):
        #to get good frame
        n = (self.focal_tracks[self.focaltrackcount]['first_frame'] 
             + len(self.focal_tracks[self.focaltrackcount]['track'])
            ) 
        if n >= len(self.files):
            self.framecount = int(len(self.files)) - 1
        else:
            self.framecount = n
        self.draw_window()