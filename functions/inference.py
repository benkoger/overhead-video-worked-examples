from detectron2.data import DatasetCatalog

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