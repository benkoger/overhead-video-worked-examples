import os
import requests
import numpy as np
import cv2
import json

from COCOJson import COCOJson


def save_labelbox_image(im_info, save_folder, overwrite=False):
    """ Save image located at adress linked to key 'Labeled Data' 
    
    Args:
        im_info: information for single annotated image in labelbox json
        save_folder: full path to folder where images should be saved
        overwrite: if True, save new image even if file already exists
        
    Returns:
        path to where image is saved
    """
    
    im_name = os.path.splitext(im_info["External ID"])[0] + ".jpg"
    image_outfile = os.path.join(save_folder, im_name)
    if os.path.exists(image_outfile):
        if not overwrite:
            return im_name
    im_bytes = requests.get(im_info['Labeled Data']).content
    im_raw = np.frombuffer(im_bytes, np.uint8)
    im = cv2.imdecode(im_raw, cv2.IMREAD_COLOR)

    cv2.imwrite(image_outfile, im)

    return im_name

def get_classes_in_json(labelbox_json, custom_class_reader=None):
    """ Get list of all unique values in labelbox json annotation.
    
    Here, classes are defined as each objects 'value.' 
    
    Args:
        labelbox_json: json file exported from labelbox
        custom_class_reader: if want to use a function to get object class
            other than annotation['value']
        
    Return list of classes
    """
    classes = set()
    for im_id, im_info in enumerate(labelbox_json):
        if im_info['Skipped']:
            continue
        for object_ann in im_info['Label']['objects']:
            if custom_class_reader:
                classes.add(custom_class_reader(object_ann))
            else:
                classes.add(object_ann['value'])
            
    return sorted(list(classes))

def labelbox_to_coco(labelbox_json_file, coco_json_file,
                     images_folder, description=None, date=None,
                     overwrite=False, verbose=False, custom_class_reader=None):
    
    """ Use labelbox json to create and save coco json and
        save corresponding annotated images.
        
        Currently just for images annotated with bounding boxes.
        
    Args:
        labelbox_json_file: path to json exported from labelbox
        coco_json_file: path to file where new coco json should
            be saved
        images_folder: path to folder were images used in labelbox
            annotations will be saved
        description: description of dataset that will be saved at
            coco_json['info']['description']
        date: date that will be saved at coco_json['info']['date']
        overwrite: if True overwrite existing image files
        verbose: if True print info like dataset classes present
        custom_class_reader: if want to use a function to get object class
            other than annotation['value']
        """
    
    coco = COCOJson(description, date)
    
    f = open(labelbox_json_file)
    labelbox_json = json.load(f)
    all_classes = get_classes_in_json(labelbox_json, custom_class_reader)
    
    # Maps class names to class ids
    label_dict = {} 
    for class_num, class_name in enumerate(all_classes):
        if verbose:
            print(class_num+1, class_name)
        coco.add_category(class_name, class_num+1)
        label_dict[class_name] = class_num + 1

    annotation_id = 1
    image_id = 1
    for im_info in labelbox_json:
        if im_info['Skipped']:
            continue
        image_name = save_labelbox_image(im_info, images_folder,
                                         overwrite=overwrite
                                        )
        image = cv2.imread(os.path.join(images_folder, image_name))
        coco.add_image(image_name, image.shape[:2], image_id)

        for annotation in im_info['Label']['objects']:
            bbox = annotation['bbox'] # ['top', 'left', 'height', 'width']
            coco_bbox = [bbox['left'], bbox['top'], bbox['width'], bbox['height']]
            if custom_class_reader:
                class_name = custom_class_reader(annotation)
            else:
                class_name = annotation['value']
            category_id = label_dict[class_name]
            coco.add_annotation_from_bounding_box(coco_bbox, image_id, 
                                                  annotation_id, category_id
                                                 )
            annotation_id += 1
        image_id += 1

    coco.write_json_file(coco_json_file)
    if verbose:
        print(f"saving at {coco_json_file}")
        print(f"{annotation_id-1} annotations from {image_id-1} images saved.")