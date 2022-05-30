from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import copy
import numpy as np
import logging
import torch

class DetectionDatasetMapper():
    """ Custom DatasetMapper.
    
    Args:
        cfg: Detectron2 config. file
        is_train: is the mapper being used for training
        calc_val_loss: same conditions as is_train=False but keeps annotations
    """
    
    def __init__(self, cfg, is_train=True, calc_val_loss=False):
        
        self.is_train = is_train
        self.calc_val_loss = calc_val_loss
        self.aug_on_test = cfg.INPUT.AUG_ON_TEST
        
        # annotations are loaded from elsewhere for testing/validation
        # So can't modify input here during testing/validation
        if cfg.INPUT.CROP.ENABLED and self.is_train:
                self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, 
                                             cfg.INPUT.CROP.SIZE)
                logging.getLogger("detectron2").info(
                    "CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None
        if cfg.INPUT.RESIZE and self.is_train:
            self.resize_gen = [T.Resize(cfg.INPUT.RESIZE_SHAPE)]
        else:
            self.resize_gen = None
                
        if self.is_train or self.aug_on_test:
            # On test, can only modify things that don't require changed annotations
            self.tfm_gens = []
            if self.is_train:
                if cfg.INPUT.VER_FLIP:
                    self.tfm_gens.append(
                        T.RandomFlip(vertical=True, horizontal=False))
                if cfg.INPUT.HOR_FLIP:
                    self.tfm_gens.append(
                        T.RandomFlip(vertical=False, horizontal=True))
            if cfg.INPUT.CONTRAST:
                self.tfm_gens.append(
                    T.RandomContrast(*cfg.INPUT.CONTRAST_RANGE))
            if cfg.INPUT.BRIGHTNESS:
                self.tfm_gens.append(
                    T.RandomBrightness(*cfg.INPUT.BRIGHTNESS_RANGE))
            if cfg.INPUT.SATURATION:
                self.tfm_gens.append(
                    T.RandomSaturation(*cfg.INPUT.SATURATION_RANGE))
        else:
            self.tfm_gens = None
            
        logger = logging.getLogger(__name__)
        logger.info("TransformGens used in training: " + str(self.tfm_gens))

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN
            )
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
            
    def _apply_image_augmentations(self, image, transforms, dataset_dict):
        """ Apply augementations to image.
        
        Args:
            image: image array
            transforms: None or already applied transforms
            dataset_dict: detectron2 dataset_dict
            
        Returns modified image, used transforms
        """
        
        if self.crop_gen:
            crop_tfm = utils.gen_crop_transform_with_instance(
                self.crop_gen.get_crop_size(image.shape[:2]),
                image.shape[:2],
                np.random.choice(dataset_dict["annotations"]),
            )
            image = crop_tfm.apply_image(image)
            if transforms:
                transforms += crop_tfm
            else:
                transforms = crop_tfm
                
        if self.resize_gen:
            image, resize_tfm = T.apply_transform_gens(self.resize_gen, image)
            if transforms:
                transforms += resize_tfm
            else:
                transforms = resize_tfm
                
        image = np.array(image) # to make writable array
                
        if self.tfm_gens:
            image, gen_transforms = T.apply_transform_gens(self.tfm_gens, image)
            if transforms:
                transforms += gen_transforms
            else:
                transforms = gen_transforms
                
        return image, transforms
    
    def _apply_annotation_augmentations(self, image, transforms, dataset_dict):
        """ Remove unneeded annotations and augment/clean those that remain.
        
        Args:
            image: image array
            transforms: None or already applied transforms
            dataset_dict: detectron2 dataset_dict
        Return dataset_dict
        """
        
        if not self.is_train and not self.calc_val_loss:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            if not self.keypoint_on:
                anno.pop("keypoints", None)

        if transforms:
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image.shape[:2], 
                    keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
        else:
            annos = [obj for obj in dataset_dict.pop("annotations")]

        instances = utils.annotations_to_instances(
            annos, image.shape[:2], mask_format=self.mask_format
        )

        # Create a tight bounding box from masks, useful when image is cropped
        if self.crop_gen and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        return dataset_dict
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code belo
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        transforms = None
        
        if self.is_train or self.aug_on_test:
            if "annotations" not in dataset_dict:
                assert False, "Must have annotations in dataset"
            image, transforms = self._apply_image_augmentations(image, 
                                                                transforms,
                                                                dataset_dict)
        dataset_dict = self._apply_annotation_augmentations(image, 
                                                            transforms,
                                                            dataset_dict)
        
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        ).contiguous()
        
        
        return dataset_dict
        
        
#             transforms = None
        
#             # Crop around an instance if there are instances in the image.
#             # USER: Remove if you don't use cropping
#             if self.crop_gen:
#                 crop_tfm = utils.gen_crop_transform_with_instance(
#                     self.crop_gen.get_crop_size(image.shape[:2]),
#                     image.shape[:2],
#                     np.random.choice(dataset_dict["annotations"]),
#                 )
#                 image = crop_tfm.apply_image(image)
#                 if transforms:
#                     transforms += crop_tfm
#                 else:
#                     transforms = crop_tfm
                
#             if self.resize_gen:
#                 image_shape = image.shape[:2]
#                 image, resize_tfm = T.apply_transform_gens(self.resize_gen, image)
#                 if transforms:
#                     transforms += resize_tfm
#                 else:
#                     transforms = resize_tfm

            
    
#             if self.tfm_gens:
#                 image, gen_transforms = T.apply_transform_gens(self.tfm_gens, image)
#                 if transforms:
#                     transforms += gen_transforms
#                 else:
#                     transforms = gen_transforms

#         image_shape = image.shape[:2]  # h, w
        
#         # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
#         # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
#         # Therefore it's important to use torch.Tensor.
#         dataset_dict["image"] = torch.as_tensor(
#             image.transpose(2, 0, 1).astype("float32")
#         ).contiguous()
        # Can use uint8 if it turns out to be slow some day

#         if not self.is_train and not self.calc_val_loss:
#             dataset_dict.pop("annotations", None)
#             dataset_dict.pop("sem_seg_file_name", None)
#             return dataset_dict

#         if "annotations" in dataset_dict:
#             # USER: Modify this if you want to keep them for some reason.
#             for anno in dataset_dict["annotations"]:
#                 if not self.mask_on:
#                     anno.pop("segmentation", None)
#                 if not self.keypoint_on:
#                     anno.pop("keypoints", None)
                    
#             # USER: Implement additional transformations if you have other types of data
#             if transforms:
#                 annos = [
#                     utils.transform_instance_annotations(
#                         obj, transforms, image_shape, 
#                         keypoint_hflip_indices=self.keypoint_hflip_indices
#                     )
#                     for obj in dataset_dict.pop("annotations")
#                     if obj.get("iscrowd", 0) == 0
#                 ]
#             else:
#                 annos = [obj for obj in dataset_dict.pop("annotations")]
                
#             instances = utils.annotations_to_instances(
#                 annos, image_shape, mask_format=self.mask_format
#             )
            

#             # Create a tight bounding box from masks, useful when image is cropped
#             if self.crop_gen and instances.has("gt_masks"):
#                 instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
#             dataset_dict["instances"] = utils.filter_empty_instances(instances)

#         return dataset_dict