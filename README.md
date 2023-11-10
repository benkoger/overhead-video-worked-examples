# Worked Examples

<p align="center">
<img src="https://github.com/benkoger/overhead-video-worked-examples/blob/main/pictures/tracks_on_map_observation088.png" height="200px">
</p>

This repository contains code and data for the worked examples described in Koger *et al.* (2023). Full descriptions of the examples are below; you can use the links there to navigate through the notebooks for each example and use the example datasets linked below to run the notebooks on existing data. Beyond using the provided data, we encourage researchers to use and modify the code to suit their own needs. If you find the code or the paper useful in your own studies, we ask that you cite this project:

Koger, B., Deshpande, A., Kerby, J.T., Graving, J.M., Costelloe, B.R., Couzin, I.D. 2023. Quantifying the movement, behaviour and environmental context of group-living animals using drones and computer vision. *Journal of Animal Ecology*, 00, 1-15. [https://doi.org/10.1111/1365-2656.13904](https://doi.org/10.1111/1365-2656.13904)

## Getting Started
### Data availability
The data required to run these examples can be downloaded from [Edmond](https://doi.org/10.17617/3.EMRZGH).

### Computing requirements 
To run the model training or inference (object detection) steps in full, the user will require an GPU which supports pytorch. This includes local NVIDIA GPUs with enough memory or most computing clusters or cloud computing services with GPU suport. We suggest a GPU with a minimum of 8GB memory (ideally 10+ GB). We provide our already trained models that researchers can use if they aren't able to train their own but want to explore the object detection step with our datasets. Additionally, storing the extracted video frames will require approximately 420 GB for the entire ungulates example video, and 45 GB for the entire gelada example video. To explore our examples one may decide to only use a clip from the video to reduce memory requirements.

The [dataset](https://doi.org/10.17617/3.EMRZGH) includes intermediate outputs for the worked examples so people interested in exploring just one part of the method can do so.

### Dependencies
To run the code in the provided notebooks, you will need to have the following packages installed:
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
These notebooks load and save data on the user's local system. Users therefore need to define local file paths where various data types can be saved and retrieved. For this, we use a single local-paths.json file for each project that all notebooks can refer to get appropriate local paths. These paths only need to be set once for each project.

Each project has a demo-local-paths.json with dummy paths as an example, the actual local-paths.json file with empty paths, and a local-paths-info.md markdown file that describes how each particular path is used. Users should edit the local-paths.json to define the necessary file paths.

## Worked Example 1: Gelada Monkeys
This example processes aerial video footage of gelada monkeys (*Theropithecus gelada*). The recordings were provided by the [Guassa Gelada Research Project](http://anthro.fullerton.edu/gelada/), and were collected between October 16, 2019 and February 28, 2020 at the Guassa Community Conservation Area in the Ethiopian highlands. Our analyses here focus on a single video observation. Geladas were recorded with a DJI Mavic 2 Pro.

In this example, we start with the raw videos and build an annotated dataset from scratch. The provided notebooks work through the steps listed below. The step numbers correspond to the step numbers in the main text and supplement of Koger *et al.* (2023).

Note that Step 2 and Step 4 require the use of 3rd party software to complete image annotation and Structure-from-Motion (SfM) tasks. We use [Labelbox](https://labelbox.com/) for annotation and [Pix4d](https://www.pix4d.com/product/pix4dmapper-photogrammetry-software) for SfM, but there are other options available.
- **Step 1: Video Recording**
    - See the paper for information on appropriate video recording techniques. We provide the example gelada video [here](https://doi.org/10.17617/3.EMRZGH).
- **Step 2: Detection**
    - **Annotation**
        - We [extract frames](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/extract_annotation_images.ipynb) from our videos that we use to build our annotated training, validation, and test set.
        - We use [Labelbox](https://labelbox.com/) with a free educational license to label these extracted frames. We annotate bounding boxes and have three classes: gelada-adult-male, gelada-other, and human-observer.
        - We [generate standard coco format json files](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/coco_from_labelbox.ipynb) from the Labelbox format .json export file that we will use to train our detection model.
            - The multiple coco format .jsons can be combined or split into separate training/validation/test files with the notebook [combine_or_split_jsons.ipynb](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/coco_from_labelbox.ipynb).
            - We can also [get stats](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/annotated_data_stats.ipynb) about our annotations including number of different classes and object sizes.
        - (Optional) After training a model (see next step), we can [use it for model assisted labeling](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/labelbox_mal.ipynb) to speed up further annotation. (Note this is just for Labelbox and the code isn't intuitive (let us know how to do it better :) )
    - **Model Training**
        - After building an annotated data set, we [train](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/train_gelada_detection.ipynb) a detection model.
        - We can [visualize the trained model's detections](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/visualize_detections.ipynb) to get an intuition of the model's performance
        - We can also [quantify the trained model's performance](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/precision-accuracy-curves.ipynb) by calculating values like precision and recall, among others.
        - If model performance isn't high enough, further annotation may help (see the previously mentioned [model assisted labeling notebook](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/labelbox_mal.ipynb))
    - **Video Processing**
        - We [extract all video frames](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/inference/extract_video_frames.ipynb) from the observation video.
            - Note: this takes a lot of memory and is unnecessary if you just want to use the detection model on this video. We do this because the frames are reused at various parts of this process.
        - We then use the trained model to [detect the geladas in each frame](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/inference/process-video.ipynb).
- **Step 3: Tracking**
    - After detecting individuals in the observation we [connect these detections together into track segments](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/tracking/detections_to_tracks.ipynb).
    - We then use a [GUI](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/tracking/track_correction_GUI.ipynb) that lets us visually connect and correct the generated track segments.
- **Step 4: Landscape Reconstruction and Geographic Coordinate Transformation**
    -  We first [extract anchor frames](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/mapping/get_anchor_frames.ipynb) from the complete observation that we use with structure from motion (SfM) software to build the 3D landscape model. The selected anchor frames are saved in a user-specified folder for SfM processing with the user’s preferred software. Unlike the ungulate worked example below, we do this without drone logs so don't have latitude, longitude, and elevation coordinates of the drone for each anchor frame.
    - We used Pix4D with an educational license for SfM processing. To follow this notebook without Pix4D, find the generated outputs [here](https://doi.org/10.17617/3.EMRZGH). For high quality geo-referencing, ground control points should be incorporated at this point in the process with your chosen SfM software. We extract GCPs from visible features in google earth. (To find the overall observation location on external maps, 10.330651 N, 39.798676 E is the location of one of the large bushes).
        - We export the generated map products into the local_paths\['pix4D_folder'\].
    - We then [calculate](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/mapping/extract_drone_movement.ipynb) how the drone (camera) moves between anchor frames. We can then optionally [confirm](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/mapping/intersegment-differences.ipynb) that the local movement estimation was accurate. Combining this local drone movement with the SfM outputs we [project the detected animal tracks into the 3D landscape](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/mapping/drone_to_landscape.ipynb) and do some initial visualization of the tracks in the landscape.
    - We can additionally [visualize](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/mapping/visualize_tracks_figure.ipynb) the georeferenced tracks on the 3D landscapes exactly as visualized in the paper. (This code is more involved and maybe less intuitive compared to the visualization previously mentioned.)
    - **Human Validation**
        - After getting tracks in coordinates of the 3D landscape (and the world if properly georeferenced), we can visually check the correspondence between animal locations in the video frames and corresponding locations in the 3D landscape.
        - For a set of random track locations, we [generate a series of crop and location files](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/mapping/map-based-validation/create-drone-to-map-validation-files.ipynb). Using a [GUI](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/mapping/map-based-validation/location_test_gui.ipynb) a human can view both crops and click the location in the 3D landscape that corresponds to the location the target animal is standing in the video.
        - Then, to [evaluate the accuracy of our location projections](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/mapping/map-based-validation/human-groundtruth-validation.ipynb), we compare the distance between the true animal location in the 3D landscape as indicated by the human and location generated by our method.

- **Step 5: Body-part Keypoint Detection**
    - In this gelada example we don't use keypoint detection. Please see the ungulates worked example for information on this step.
- **Step 6: Landscape Quantification**
    - Many important landscape features, like ground topology, elevation, and color, are already quantified during the structure from motion step. For examples of extracting more complex landscape features, see our example extracting possible game trails from the landscape in the ungulates worked examples below.
To work through this example in sequence, [download the data](https://doi.org/10.17617/3.EMRZGH) and start [here](https://github.com/benkoger/overhead-video-worked-examples/blob/main/geladas/detection/model-training/extract_annotation_images.ipynb).

## Worked Example 2: Kenyan Ungulates

This example processes aerial video footage of African ungulate herds. We recorded ungulate groups at [Ol Pejeta](https://www.olpejetaconservancy.org/) and [Mpala Conservancies](https://mpala.org/) in Laikipia, Kenya over two field seasons, from November 2 to 16, 2017 and from March 30 to April 19, 2018. In total, we recorded thirteen species, but here we focus most of our analyses on a single 50-minute observation of a herd of 18 Grevy’s zebras (*Equus grevyi*). We used DJI Phantom 4 Pro drones, and deployed two drones sequentially in overlapping relays to achieve continuous observations longer than a single battery duration.

In this example, we start with a pre-annotated dataset (we previously annotated it with now-outdated software). For an example of building an annotated dataset from scratch, please see the gelada example. Our annotated image set contains five classes: zebra, impala, buffalo, waterbuck, and other spanning 1913 annotated video frames. See the main text of the paper or [annotated_data_stats.ipynb](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/detection/model-training/annotated_data_stats.ipynb) for more details on the annotated dataset.

The provided notebooks work through the steps listed below. The step numbers correspond to the step numbers in the main text and supplement of Koger *et al.* (2023).

Note that Step 4 requires the use of 3rd party software to complete Structure-from-Motion tasks. We use [Pix4D](https://www.pix4d.com/product/pix4dmapper-photogrammetry-software), but there are other options available.
- **Step 1: Video Recording**
    - See the paper for information on appropriate video recording techniques. We provide the videos from the example Grevy's zebra observation [here](https://doi.org/10.17617/3.EMRZGH).
- **Step 2: Detection**
    - We start by [training](./ungulates/detection/model-training/train_ungulate_detection.ipynb) and then [evaluating](./ungulates/detection/model-training/precision-accuracy-curves.ipynb) a model that detects various ungulate species. We then use this model to [process an observation of Grevy's zebras](./ungulates/detection/inference/process-video.ipynb) that spans three overlapping drone flights.
    - Before processing, we [extract the individual video frames](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/detection/inference/extract_video_frames.ipynb) from the complete observation. These will be used throughout the pipeline, including during detection.
- **Step 3: Tracking**
    - After detecting individuals in the observation we [connect these detections together into track segments](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/tracking/detections_to_tracks.ipynb).
    - We then use a [GUI](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/tracking/track_correction_GUI.ipynb) that lets us visually connect and correct the generated track segments.
- **Step 4: Landscape Reconstruction and Geographic Coordinate Transformation**
    -  We first [extract anchor frames](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/get_anchor_frames.ipynb) from the complete observation that we use with structure from motion (SfM) software to build the 3D landscape model. The selected anchor frames are saved in a user-specified folder for SfM processing with the user’s preferred software. The latitude, longitude, and elevation coordinates of the drone for each anchor frame as recorded in the drone logs is saved as a .csv in the input format used by Pix4D.
    - We used Pix4D with an educational license for SfM processing. To follow this notebook without Pix4D, find the generated outputs [here](https://doi.org/10.17617/3.EMRZGH). For high quality geo-referencing, ground control points should be incorporated at this point in the process with your chosen SfM software.
    - We then [calculate](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/extract_drone_movement.ipynb) how the drone (camera) moves between anchor frames. We can then optionally [confirm](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/intersegment-differences.ipynb) that the local movement estimation was accurate. Combining this local drone movement with the SfM outputs we [project the detected animal tracks into the 3D landscape](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/drone_to_landscape.ipynb).
    - We can [visualize](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/visualize_tracks_figure.ipynb) the georeferenced tracks on the 3D landscapes.

- **Step 5: Body-part Keypoint Detection**
    - We [extract square crops](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/postures/crop_out_tracks.ipynb) around each tracked individual. These crops can be used with one of many existing open source animal keypoint tracking softwares such as [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut), [SLEAP](https://sleap.ai/), or [DeepPoseKit](https://github.com/jgraving/DeepPoseKit).
    - We [provide](https://doi.org/10.17617/3.EMRZGH) complete keypoints for the observation previously generated by DeepPoseKit ([performance stats](https://elifesciences.org/articles/47994)) as well as a [page](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/postures/instructions_for_DeepLabCut.md) describing how to use DeepLabCut to train a new model to use for keypoint detection in the context of this method.

- **Step 6: Landscape Quantification**
    - Many important landscape features, like ground topology, elevation, and color, are already quantified during the structure from motion step.
    - For demonstration purposes, we also include a notebook for [training](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/game-trails(demo)/train_gametrail_detection.ipynb) a CNN to detect game trails in the landscape and another notebook for [using this model](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/game-trails%28demo%29/gametrail_inference.ipynb) on our 3D landscape maps.
    - This is just meant as a demonstration of what is possible and hasn't been carefully validated beyond visual inspection
        - See the notebooks for details on the training data and training regime used.


To work through this example in sequence, [download the data](https://doi.org/10.17617/3.EMRZGH) and start [here](./ungulates/detection/model-training/train_ungulate_detection.ipynb).
