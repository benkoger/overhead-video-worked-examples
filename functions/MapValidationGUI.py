import cv2
import numpy as np
import random
import matplotlib.cm as cm



def rescale_image(image, max_size):
    """ Rescale image so that longest dimension equals max size.
    
    Args:
        image: 3d numpy array
        max_size: number longest output size of image
    """
    
    im_size = np.array(image.shape)
    longest_side = np.max(im_size)
    scale = max_size / longest_side
    new_im_shape = (im_size * scale).astype(int)
    new_image = cv2.resize(image, (new_im_shape[1], new_im_shape[0]))
    
    return new_image, scale


class MapValidationGUI():
    
    #constructor
    def __init__(self, map_image_files, drone_frame_files, frame_positions_files, map_positions_files, max_size):
        """ 
        Args:
            map_image_files: list of image files of landscape map 
            drone_frame_files: list of image files from drone video
            frame_positions_files: list files of locations of individuals in drone frames
            map_posiotions_files: list of files of arrays that already have xy offset 
                and points will be added to
            max_size: largest dimension in pixels of output display
        """
        assert len(map_image_files) == len(drone_frame_files)
        assert len(drone_frame_files) == len(frame_positions_files)
        self.num_test = len(map_image_files)
        
        self.test_num = 0
        self.focal_ind = 1 # focal individual in frame
        
        self.map_image_files = map_image_files
        self.drone_frame_files = drone_frame_files
        self.frame_positions_files = frame_positions_files
        self.map_positions_files = map_positions_files
        
        self.max_size = max_size
        self.show_points = True
        
        self.load_new_test()
        
    def load_new_test(self):
        raw_map_image = cv2.imread(self.map_image_files[self.test_num]) 
        raw_drone_frame = cv2.imread(self.drone_frame_files[self.test_num]) 
        self.raw_map_image, self.map_rescale = rescale_image(raw_map_image, self.max_size)
        self.raw_drone_frame, self.frame_rescale = rescale_image(raw_drone_frame, self.max_size)
        
        self.frame_positions = np.load(self.frame_positions_files[self.test_num])
        self.map_positions = np.load(self.map_positions_files[self.test_num])
        self.colors = [] 
        for ind in range(len(self.map_positions)):
            color = cm.jet(ind/len(self.map_positions))
            color = [int(c*255) for c in color]
            self.colors.append(color)
        self.focal_ind = 1
        
    def refresh_windows(self):
        self.map_image = np.copy(self.raw_map_image)
        self.drone_frame = np.copy(self.raw_drone_frame)
        
        
    def save_map_positions(self):
        """ Overwrite test's map positions file with current values."""
        
        np.save(self.map_positions_files[self.test_num], self.map_positions)
        
    def change_frame(self, amount):
        """Change test frame forward or backward by amount.
        0 is minimum frame ind and number of frames is max (no periodic boundaries)
        
        Args:
            amount: number of frames to move positive or negative"""
        
        self.save_map_positions()
        
        self.test_num += amount
        self.test_num = np.max([0, self.test_num])
        self.test_num = np.min([len(self.map_image_files)-1, self.test_num])
        self.load_new_test()
        
    def change_focal_ind(self, amount):
        """Change focal individual forward or backward by amount.
        1 is minimum focal ind and number of individuals is max (no periodic boundaries)
        
        Args:
            amount: number of indexes to change positive or negative"""
        
        self.focal_ind += amount
        self.focal_ind= np.max([1, self.focal_ind]) # 1 because 0 is info about min x y
        
        self.focal_ind = np.min([len(self.frame_positions)-1, self.focal_ind])
        
        
    def draw_frame_positions(self):
        """ Draw the location of all the animals in the video frame."""
        if self.show_points:
            for ind, (position, color) in enumerate(zip(self.frame_positions[1:], self.colors[1:])):
                cv2.circle(self.drone_frame, 
                           (int(position[0]*self.frame_rescale), int(position[1]*self.frame_rescale)), 
                           radius=int(.006*self.max_size), color=color, thickness=-1)
                if ind+1 == self.focal_ind:
                    cv2.circle(self.drone_frame, 
                           (int(position[0]*self.frame_rescale), int(position[1]*self.frame_rescale)), 
                           radius=int(.06*self.max_size), color=color, thickness=1)
            
    def draw_map_positions(self):
        """ Draw the estimated location of all the animals in the map."""
        
        if self.show_points:
            for position, color in zip(self.map_positions[1:], self.colors[1:]): # ind 0 relates values back to original frame
                if np.all(position!=-1):
                    cv2.circle(self.map_image, 
                               (int(position[0]*self.map_rescale), int(position[1]*self.map_rescale)), 
                               radius=int(.006*self.max_size), color=color, thickness=-1)
            
            
    def change_color(self, color_index, color):
        """ Change the color of one of the individuals.
        
        Args:
            color_index: index to change
            color: new color (brg)"""
        
        self.colors[color_index] = color
        
        
    def not_present(self):
        # Current ind is not in map area
        self.map_positions[self.focal_ind] = np.array([-2, -2])
        
    def add_map_point(self, x, y):
        """Record x and y position of click in map image assosiated with focal ind.
        Args:
            x, y: from mouse click
        """
        self.map_positions[self.focal_ind] = np.array([x/self.map_rescale, y/self.map_rescale])
        
        self.change_focal_ind(1)
        
        
        
    def show_windows(self):
        cv2.imshow('map_image', self.map_image)
        cv2.imshow('video_frame', self.drone_frame)
        
    def react_to_keypress(self, key):
        """ Process key press
        Args:
            key: return from cv2 cv2.waitkey
        """
        
        if key == ord('l'):
            self.change_frame(1)
        elif key == ord('j'):
            self.change_frame(-1)
        elif key == ord('i'):
            self.change_focal_ind(1)
        elif key == ord('k'):
            self.change_focal_ind(-1)
        elif key == ord('p'):
            self.toggle_show_points()
        elif key == ord('n'):
            self.not_present()
            
    def clicked(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_map_point(x,y)
            
    def toggle_show_points(self):
        self.show_points = not self.show_points
            
        
        
        
        
        
        
        
        
    
