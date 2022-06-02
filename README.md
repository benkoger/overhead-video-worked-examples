# Worked Examples

<p align="center">
<img src="https://github.com/benkoger/overhead-video-worked-examples/blob/main/pictures/tracks_on_map_observation088.png" height="320px">
</p>

This repository contains code and data for the worked examples described in Koger *et al.* (n.d.). Full descriptions of the examples are below; you can use the links there to navigate through the notebooks for each example. We encourage users to download the notebooks and modify the code to suit their own needs. If you find the code or the paper useful in your own studies, we ask that you cite this project. *insert citation information for paper and for code (if different)*

## Getting Started
### Computing requirements
Implementing this method as here requires specific computing resources that exceed many personal computers. Computing requirements can be met either with certain high performance gaming-type machines or using dedicated computing clusters including those based in the cloud.  The large CNNs used here are designed to run on graphical processing units (GPUs) or specialized deep learning processors (for example tensor processing units (TPUs). Other types of models not implemented here, however, such as mobilenet (https://arxiv.org/abs/1704.04861), could be used on a CPU.  Among GPUs, many deep learning frameworks only work with those designed by the NVIDIA [NVIDIA Corporation, USA] although this is changing. While specific GPU memory requirements depend on the exact model and images, at least 8GB but ideally 10+ GB of memory is necessary. SfM mapping tasks, on the other hand, are generally limited by a computer’s RAM requiring up to 16GB for mapping projects with 500-1000 images (https://support.pix4d.com/hc/en-us/articles/115002439383-Computer-requirements-PIX4Dmapper). Lastly, storing the high resolution videos and the individual frames that must be extracted from them can require 100s of gigabytes or terabytes of hard drive space. For the worked examples specifically, the exrtacted frames from the ungulates worked examples takes around 420 gigabytes of space, while the gelada frames take 45 gigabytes of space.

We provide intermediate outputs for the worked examples at *data storage* so people intersted in exploring just one part of the method can do so.

### Dependencies
- [cv2](https://opencv.org/)
- [Detectron2](https://detectron2.readthedocs.io/en/latest/)
- [fvcore](https://github.com/facebookresearch/fvcore)
- [gdal](https://gdal.org/)
- [imutils](https://github.com/PyImageSearch/imutils)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [pycocotools](https://pypi.org/project/pycocotools/)
- [requests](https://pypi.org/project/requests/)
- [scipy](https://scipy.org/)
- [tabulate](https://pypi.org/project/tabulate/)
- [torch](https://pytorch.org/docs/stable/torch.html)
- [utm](https://pypi.org/project/utm/)
- [yaml](https://pyyaml.org/wiki/PyYAMLDocumentation)

### Local file paths
The code requires users to specify various locations on their local machines where files are stored or should be saved. These filepaths are saved in a .json file titled 'local-paths.json', (LOCATED WHERE? - CURRENTLY MINE IS SAVED IN THE UNGULATES FOLDER, BUT SHOULD PROBABLY BE SAVED IN BASE FOLDER - overhead-video-worked-examples - IF IT IS TO BE USED FOR BOTH WORKED EXAMPLES). This file can be edited in a standard text editor and should specify the following information:
- "general_dection_path": *insert description*
- "annotations_folder": *insert description*
- "labelbox_folder": *insert description*
- "videos_folder": *insert description*
- "base_ungulates": *insert description*
- "half_res_base_ungulates": *insert description*
- "labelbox_key": *insert description*
- "model_folder": *insert description*
- "detectron_path": *insert description*
- "overhead_functions_path": *insert description*
- "base_data_path": *insert description*
- "base_frames_folder": *insert description*
- "pix4d_folder": *insert description*
- "processed_folder": *insert description*

## Worked Example 1: Kenyan Ungulates

This example processes aerial video footage of African ungulate herds. We recorded ungulate groups at Ol Pejeta and Mpala Conservancies in Laikipia, Kenya over two field seasons, from November 2 to 16, 2017 and from March 30 to April 19, 2018. In total, we recorded thirteen species, but here we focus most of our analyses on Grevy’s zebra (*Equus grevyi*). We used DJI Phantom 4 Pro drones, and deployed two drones sequentially in overlapping relays to achieve continuous observations longer than a single battery duration.

Our annotated image set contains five classes: zebra, impala, buffalo, waterbuck, and other spanning 1913 annotated video frames. See the main text of the paper or [annotated_data_stats.ipynb](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/detection/model-training/annotated_data_stats.ipynb) for more details on the annotated dataset.

In this example, we start with a pre-annotated dataset (we previously annotated it with now outdated software). For an example of building an annotated dataset from scratch, please see the gelada example below. The provided notebooks work through the steps listed below. The step numbers correspond to the step numbers in the main text and supplement of Koger *et al.* (n.d.). 
- **Step 1: Video Recording** 
    - See the paper for information on proper video recording. We provide all videos from a complete ~45 minute observation of Grevy's zebra [here](data repo).
- **Step 2: Detection** 
    - We start by [training](./ungulates/detection/model-training/train_ungulate_detection.ipynb) and then [evaluating](./ungulates/detection/model-training/precision-accuracy-curves.ipynb) a model that detects various ungulate species. We then use this model to [process an observation of Grevy's zebras](./ungulates/detection/inference/process-video.ipynb) that spans three overlapping drone flights. 
    - Before processing, we [extract the individual video frames](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/detection/inference/extract_video_frames.ipynb) from the complete observation. These will be used throughout the process, including during detection. 
- **Step 3: Tracking** 
    - After detecting individuals in the observation we [connect these detections together into track segments](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/tracking/detections_to_tracks.ipynb). 
    - We then use a [GUI](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/tracking/track_correction_GUI.ipynb) that lets us visually connect and correct the generated track segments.
- **Step 4: Landscape Reconstruction and Geographic Coordinate Transformation** 
    -  We first [extract anchor frames](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/get_anchor_frames.ipynb) from the complete observation that we use with structure from motion (SfM) software to build the 3D landscape model. The selected anchor frames are saved in a user-specified folder for structure from motion processing with the user’s chosen software. The latitude, longitude, and elevation coordinates of the drone for each anchor frame as recorded in the drone logs is saved as a .csv in the input format used by Pix4D. 
    - We used Pix4D with an educational license for SfM processing. To follow this notebook without Pix4D, find the generated outputs [here](data repo). For high quality geo-referencing ground control points should be incorporated at this point in the process with your chosen SfM software. 
    - We then [calculate](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/extract_drone_movement.ipynb) how the drone (camera) moves between anchor frames. We can then optionally [confirm](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/intersegment-differences.ipynb) that the local movement estimation was accurate. Combining this local drone movement with the SfM outputs we [project the detected animal tracks into the 3D landscape](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/drone_to_landscape.ipynb). 
    - We can [visualize](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/visualize_tracks_figure.ipynb) the georeferenced tracks on the 3D landscapes. 

- **Step 5: Body-part Keypoint Detection** 
    - We [extract square crops](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/postures/crop_out_tracks.ipynb) around each tracked individual. These crops can be used with one of many existing open source animal keypoint tracking softwares such as [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut), [SLEAP](https://sleap.ai/), or [DeepPoseKit](https://github.com/jgraving/DeepPoseKit). 
    - We [provide](data repo) complete keypoints for the observation previously generated by DeepPoseKit ([performance stats](https://elifesciences.org/articles/47994)) as well as a [page]() describing how to use DeepLabCut to train a new model to use for keypoint detection in the context of this method.

- **Step 6: Landscape Quantification** - 

Note that Step 4 requires the use of 3rd party software to complete Structure-from-Motion tasks. We use [Pix4D](https://www.pix4d.com/product/pix4dmapper-photogrammetry-software), but there are other options available. 