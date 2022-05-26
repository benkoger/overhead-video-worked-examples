from detectron2.data import DatasetCatalog
from torchvision.ops import nms

def load_inference_instances(image_files):
    """ Use all given files to make Detectron2 format list of dicts."""
    dataset_dicts = []
    for image_file in image_files:
        record = {"file_name": image_file}
        dataset_dicts.append(record)
    return dataset_dicts

def register_inference_instances(name, image_files):
    """ Register dataset to use for inference. 
    
    Only image file paths are used.
    
    Args:
        name: name of Dataset
        image_files: list of image_files
    """
    DatasetCatalog.register(name, lambda: load_inference_instances(image_files))

def nms_all_classes(instances, iou_thresh):
    """ Apply non-maximum suppression to inference instances regardless of class.
    
    Args:
        instances: instances from detectron2 model
        iou_thresh: threshold to use for nms
        
    returns resulting instances after nms is applied
    """
    valid_ind = nms(instances.pred_boxes.tensor, instances.scores, iou_thresh)
    instances.pred_boxes.tensor = instances.pred_boxes.tensor[valid_ind]
    instances.scores = instances.scores[valid_ind]
    instances.pred_classes = instances.pred_classes[valid_ind]
    
    return instances