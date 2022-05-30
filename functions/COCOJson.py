import json
import shutil
import numpy as np
import cv2


def get_bbox(contour):
    """ Get bounding box of contour.
    
    Args:
        contour: opencv contour object
        
    Returns:
        Upper left corner x, y and bbox width and height as list
    """
    
    upper_left = np.min(contour, 0)
    bottom_right = np.max(contour, 0)
    bbox_xy = (bottom_right - upper_left)
    
    return [int(upper_left[0]), int(upper_left[1]), int(bbox_xy[0]), int(bbox_xy[1])]

class COCOJson:
    
    
    def __init__(self, description, date_created):
        """ Info to create coco.json annotation file. 
    
        Args:
            description: string, description of dataset
            date_created: string, date dataset created

        """
        self.coco_dict = {}
        self.create_boilerplate(description, date_created)
        
        
    def create_boilerplate(self, description='', date_created=''):
        """ Creates the generic peices of coco annotation and returns dict.
        
        Args:
            description: string, general description of the dataset
            date_created: string, date the dataset is created
        """
        

        self.coco_dict['info'] = []
        self.coco_dict['info'].append({
            'description': description,
            'url': '',
            'version': '1.0',
            'year': 2020,
            'contributor': 'Ben Koger',
            'date_created': date_created    
        })

        self.coco_dict['licenses'] = []
        self.coco_dict['licenses'].append({
            'url': '',
            'id': 0,
            'name': '',  
        })
        
        self.coco_dict['images'] = []
        self.coco_dict['annotations'] = []
        self.coco_dict['categories'] = []
        
        
    def add_image(self, file_name, image_shape, image_id):
        """ Add image to coco dict.
        
        Args:
            file_name: generally basename of full path to image
            image_shape: shape of the image (height, width)
            image_id: int, should be unique for each image in coco dataset (starting at 1)
            
        """

        self.coco_dict['images'].append({
            'license': 0,
            'file_name': file_name,
            'coco_url': '',
            'height': image_shape[0],
            'width': image_shape[1],
            'date_captured': '',
            'flickr_url': '',
            'id': image_id
            })
    
    def add_category(self, name, category_id, supercategory=''):
        """ Add new category to dataset.
        
        Args:
            name: string, name of new category
            category_id: category id, should be unique for each 
                category in dataset (starting at 1)
            supercategory: string, supercategory, if any
        
        """
        
        self.coco_dict['categories'].append({
            'supercategory': supercategory,
            'id': category_id,
            'name': name
            })
        
    def add_annotation_from_bounding_box(self, bbox, image_id, annotation_id, category_id):
        """ Add new annotation from bounding box to dataset.
        
        Args:
            bbox: [upper_left_x, upper_left_y, bbox_height, bbox_width]
            image_id: id of the image the annotation came from
            annotation_id: id number of this annotation, should 
                be unique for each annotation in dataset (from 1)
            category_id: category id, should be unique for each 
                category in dataset (starting at 1)
        """
        
        segmentation = [bbox[0], bbox[1],
                        bbox[0] + bbox[2], bbox[1],
                        bbox[0] + bbox[2], bbox[1] + bbox[3],
                        bbox[0], bbox[1] + bbox[3]]
        
        self.coco_dict['annotations'].append({
                'segmentation': [segmentation],
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': bbox,
                'category_id': category_id,
                'id': annotation_id
            })
        
        
    
        
    def add_annotation_from_contour(self, contour, image_id, annotation_id, category_id=1):
        """ Add annotation from openCV contour object.
        
        Args:
            contour: openCV contour object
            image_id: id of the image the contour came from
            annotation_id: id number of this annotation, should 
                be unique for each annotation in dataset (from 1)
        
        """
        segmentation = list(contour.reshape(-1))
        segmentation = [int(val) for val in segmentation]
        
        if len(segmentation) % 2 == 0 and len(segmentation) >= 6:
            # must have at least three points each with an x and a y
            self.coco_dict['annotations'].append({
                'segmentation': [segmentation],
                'area': cv2.contourArea(contour),
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': get_bbox(contour),
                'category_id': category_id,
                'id': annotation_id
            })
        else:
            print("removing invalid segmentation...")
        
    def _get_contours_from_binary(self, binary_image):
        """ Get all contours from binary image.
        
        Args:
            binary_image: 2d array of zeros and ones"""
        
        contours, hierarchy = cv2.findContours(binary_image.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [np.squeeze(contour) for contour in contours]
        
        return contours, hierarchy    
    
        
    def add_annotations_from_binary_mask(self, binary_image, image_id, current_annotation_id, category_id):
        """ Add every contour in binary image as seperate annotation.
        
        Args:
            binary_image: 2d array of ones and zeros
            image_id: id of the image that this binary image is the label for
            current_annotation_id: the next annotation id value to use in the dataset
            category_id: the annotation category of the binary mask
        """
        
        contours, hierarchy = self._get_contours_from_binary(binary_image)
        
        if not contours:
            return current_annotation_id
        
        for contour, h in zip(contours, hierarchy[0]):
            if len(contour.shape) < 2:
                # Only single point
                continue
#             if h[3] != -1:
#                 self.add_annotation_from_contour(contour, image_id, current_annotation_id, category_id=2)
#             else:
            self.add_annotation_from_contour(contour, image_id, current_annotation_id, category_id)
            current_annotation_id += 1
        
        return current_annotation_id
        
        
        
        
    def copy_image_to_image_folder(self, outfolder, file):
        """ Copy image to an annotated images folder.
        
        Args:
            outfolder: path of folder to copy images to
            file: existing file of image that should be copied"""
        
        shutil.copy(file, outfolder)
        
        
    def write_json_file(self, outfile):
        """ Write the information in coco_dict to json file.
        
        Args:
            outfile: string, file that the json file will be written to
            
        """
        
        with open(outfile, 'w') as outfile:
            json.dump(self.coco_dict, outfile, indent=4)
            
    def add_keypoint(self, x, y, image_id, annotation_id, category_id=1):
        """ Add keypoint annotation.
        
        Args:
            x: keypoint x value
            y: keypoint y value
            image_id: id of the image the contour came from
            annotation_id: id number of this annotation, should 
                be unique for each annotation in dataset (from 1)
            category_id: id of the annotation category
        
        """

        self.coco_dict['annotations'].append({
            'segmentation': [[]],
            'area': [],
            'iscrowd': 0,
            'keypoints': [x, y, 2],
            'num_keypoints': 1,
            'image_id': image_id,
            'bbox': [],
            'category_id': category_id,
            'id': annotation_id
        })