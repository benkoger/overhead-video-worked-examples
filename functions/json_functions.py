import copy
import json
import copy
import os
import numpy as np

def create_empty_annotation_json(json_dict):
    """ Only preserve dataset level info"""
    
    new_dict = copy.deepcopy(json_dict)
    new_dict['images'] = []
    new_dict['annotations'] = []
    return new_dict

def get_annotations_for_id(annotation_dicts, image_id):
    """ Get all annotations that go with a given image id.
    
    Args:
        annotation_dicts: val stored in coco dataset under annotation key
        image_id: image id that you want annotations for
        
    Return annotations
    """
    annotations = []
    for annotation_dict in annotation_dicts:
        if annotation_dict['image_id'] == image_id:
            annotations.append(copy.deepcopy(annotation_dict))
    return(annotations)


def create_train_val_split(json_file, fraction_val, save_folder=None,
                           train_name="train.json", val_name="val.json"):
    """
        Args:
            json_file: full path to json file for all annotations
            fraction_val: fraction of total dataset should be used for
                testing (.25 -> a quarter of total used for testing)
            save_folder: path to folder to save new .json files.
                If None, then save in same file as current json
    """
    
    with open(json_file, "r") as read_file:
        json_dict = json.load(read_file)
        
    print('There are {} annotated images.'.format(
        len(json_dict['images'])))
    
    # image ids to use
    image_ids = np.arange(len(json_dict['images']))
    
    train_dict = create_empty_annotation_json(json_dict)
    val_dict = create_empty_annotation_json(json_dict)

    images = sorted([an for an in json_dict['images']], key=lambda an: an['id']) 

    images = [images[image_id] for image_id in image_ids]

    images_added = 0

    for image_num, image_dict in enumerate(images):
        image_id = image_dict['id']
        new_annotations = get_annotations_for_id(json_dict['annotations'], image_id)
        if len(new_annotations) != 0:
            if images_added % int(1/fraction_val) == 0:
                # validation image
                val_dict['images'].append(image_dict)
                val_dict['annotations'].extend(new_annotations)
            else:
                # training image
                train_dict['images'].append(image_dict)
                train_dict['annotations'].extend(new_annotations)
            images_added += 1

    # correct annotation ids
    for new_id, _ in enumerate(train_dict['annotations']):
        train_dict['annotations'][new_id]['id'] = new_id + 1
    for new_id, _ in enumerate(val_dict['annotations']):
        val_dict['annotations'][new_id]['id'] = new_id + 1

    print('{} training images with {} annotations.'.format(
        len(train_dict['images']),len(train_dict['annotations'])))
    print('{} validation images with {} annotations.'.format(
        len(val_dict['images']),len(val_dict['annotations'])))

    save_folder = os.path.dirname(json_file)

    with open(os.path.join(save_folder, train_name), "w") as write_file:
        json.dump(train_dict, write_file, indent=4, separators=(',', ': '))

    with open(os.path.join(save_folder, val_name), "w") as write_file:
        json.dump(val_dict, write_file, indent=4, separators=(',', ': '))
        
def get_annotations_based_on_id(annotation_dicts, image_id, 
                                new_id, annotation_id):
    annotations = []
    for annotation_dict in annotation_dicts:
        if annotation_dict['image_id'] == image_id:
            annotations.append(copy.deepcopy(annotation_dict))
            annotations[-1]['image_id'] = new_id
            annotations[-1]['id'] = annotation_id 
            annotation_id += 1
    return(annotations, annotation_id)

def combine_jsons(json_files, out_file=None):
    """ Combine multiple JSON file into a new single consistent JSON file.
    
    Args:
        json_files (list): list of json file strings
        out_file (string): full path of file where we want to save new file
            if None, don't save
    
    Return combined json
    """
    
    json_dicts = []

    for json_file in json_files:
        with open(json_file, "r") as read_file:
            json_dict = json.load(read_file)
            json_dicts.append(json_dict)
    
    total_images = 0
    for json_dict in json_dicts:
        total_images += len(json_dict['images'])
        
    print('There are {} annotated images in the JSON files.'.format(
        total_images))
    
    new_dict = create_empty_annotation_json(json_dicts[0])

    images_added = 0
    annotation_id = 0

    for json_dict in json_dicts:

        images = [an for an in json_dict['images']]
        images = sorted(images, key=lambda an: an['id']) 

        for image_num, image_dict in enumerate(images):
            image_id = image_dict['id']
            new_annotations, annotation_id = get_annotations_based_on_id(
                json_dict['annotations'], image_id, images_added+1, annotation_id)
            if len(new_annotations) != 0:
                new_dict['images'].append(image_dict)
                new_dict['images'][-1]['id'] = images_added + 1
                new_dict['annotations'].extend(new_annotations)
                images_added += 1

    print('{} images added to new .json'.format(len(new_dict['images'])))
    print('{} annotations added to new .json'.format(len(new_dict['annotations'])))
    
    if out_file:
        with open(out_file, "w") as write_file:
            json.dump(new_dict, write_file, indent=4, 
                      separators=(',', ': '))
    else:      
        return new_dict
