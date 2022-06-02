### DeepLabCut (as well as SLEAP) is a currently well maintained software package for animal keypoint detection. 

In this example, we use data previously annotated using DeepPosekit ((Graving et al., 2019)). The data can be found at https://github.com/jgraving/DeepPoseKit-Data/tree/master/datasets/zebra. It contains annotations for nine body parts (snout, head, neck, right and left shoulders, right and left hindquarters, tail base, tail tip) for a randomly selected subset of cropped images of individual zebras (n= 2140 ) in HDF5 format.

Follow the instructions in the notebook in this folder called 'data_conversion_DPK_to_DLC.ipynb' to start a DeepLabCut project, convert our annotated data to the proper format, and use the resulting data with their interface.

Crops extracted from the full observation extracted with 'crop_out_tracks.ipynb' can be fed to DeepLabCut to extract keypoints for all individuals.