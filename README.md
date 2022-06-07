# Worked Examples

<p align="center">
<img src="https://github.com/benkoger/overhead-video-worked-examples/blob/main/pictures/tracks_on_map_observation088.png" height="200px">
</p>

This repository contains code and data for the worked examples described in Koger *et al.* (n.d.). Full descriptions of the examples are below; you can use the links there to navigate through the notebooks for each example. We encourage users to download the notebooks and modify the code to suit their own needs. If you find the code or the paper useful in your own studies, we ask that you cite this project:

Koger, B., Deshpande, A., Kerby, J.T., Graving, J.M., Costelloe, B.R., Couzin, I.D. Multi-animal behavioral tracking and environmental reconstruction using drones and computer vision in the wild.

## Getting Started
### Computing requirements
To run these examples in full, the user will require an NVIDIA GPU. We suggest a GPU with a minimum of 8GB memory (ideally 10+ GB). *maybe a note about which steps specifically require the GPU, so it's clear that they can still play around with the code even if they don't have the GPU?* Additionally, storing the extracted video frames will require approximately 420 GB for the ungulates example, and 45 GB for the gelada example.

We provide intermediate outputs for the worked examples at *data storage* so people interested in exploring just one part of the method can do so.

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
This example processes aerial video footage of gelada monkeys (*Theropithecus gelada*). The recordings were provided by the Guassa Gelada Research Project, and were collected between October 16, 2019 and February 28, 2020 at the Guassa Community Conservation Area in the Ethiopian highlands. Our analyses here focus on a single video observation. Geladas were recorded with a DJI Mavic 2 Pro.

In this example, we start with the raw videos and build an annotated dataset from scratch. The provided notebooks work through the steps listed below. The step numbers correspond to the step numbers in the main text and supplement of Koger *et al.* (n.d.).

Note that Step 2 and Step 4 require the use of 3rd party software to complete image annotaiton and Structure-from-Motion (SfM) tasks. We use [Labelbox](https://labelbox.com/) for annotation and [Pix4d](https://www.pix4d.com/product/pix4dmapper-photogrammetry-software) for SfM, but there are other options available.
- **Step 1: Video Recording**
    - See the paper for information on appropriate video recording techniques. We provide the example gelada video here (*link to data repo*).
- **Step 2: Detection**
- **Step 3: Tracking**
- **Step 4: Landscape Reconstruction and Geographic Coordinate Transformation**
- **Step 5: Body-part Keypoint Detection**
- **Step 6: Landscape Quantification**
To work through this example in sequence, download the data (insert link to dataset) and start [here]().

## Worked Example 2: Kenyan Ungulates

This example processes aerial video footage of African ungulate herds. We recorded ungulate groups at Ol Pejeta and Mpala Conservancies in Laikipia, Kenya over two field seasons, from November 2 to 16, 2017 and from March 30 to April 19, 2018. In total, we recorded thirteen species, but here we focus most of our analyses on a single ~45-minute observation of a herd of 18 Grevy’s zebras (*Equus grevyi*). We used DJI Phantom 4 Pro drones, and deployed two drones sequentially in overlapping relays to achieve continuous observations longer than a single battery duration.

In this example, we start with a pre-annotated dataset (we previously annotated it with now-outdated software). For an example of building an annotated dataset from scratch, please see the gelada example. Our annotated image set contains five classes: zebra, impala, buffalo, waterbuck, and other spanning 1913 annotated video frames. See the main text of the paper or [annotated_data_stats.ipynb](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/detection/model-training/annotated_data_stats.ipynb) for more details on the annotated dataset.

The provided notebooks work through the steps listed below. The step numbers correspond to the step numbers in the main text and supplement of Koger *et al.* (n.d.).

Note that Step 4 requires the use of 3rd party software to complete Structure-from-Motion tasks. We use [Pix4D](https://www.pix4d.com/product/pix4dmapper-photogrammetry-software), but there are other options available.
- **Step 1: Video Recording**
    - See the paper for information on appropriate video recording techniques. We provide the videos from the example Grevy's zebra observation [here](data repo).
- **Step 2: Detection**
    - We start by [training](./ungulates/detection/model-training/train_ungulate_detection.ipynb) and then [evaluating](./ungulates/detection/model-training/precision-accuracy-curves.ipynb) a model that detects various ungulate species. We then use this model to [process an observation of Grevy's zebras](./ungulates/detection/inference/process-video.ipynb) that spans three overlapping drone flights.
    - Before processing, we [extract the individual video frames](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/detection/inference/extract_video_frames.ipynb) from the complete observation. These will be used throughout the pipeline, including during detection.
- **Step 3: Tracking**
    - After detecting individuals in the observation we [connect these detections together into track segments](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/tracking/detections_to_tracks.ipynb).
    - We then use a [GUI](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/tracking/track_correction_GUI.ipynb) that lets us visually connect and correct the generated track segments.
- **Step 4: Landscape Reconstruction and Geographic Coordinate Transformation**
    -  We first [extract anchor frames](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/get_anchor_frames.ipynb) from the complete observation that we use with structure from motion (SfM) software to build the 3D landscape model. The selected anchor frames are saved in a user-specified folder for SfM processing with the user’s preferred software. The latitude, longitude, and elevation coordinates of the drone for each anchor frame as recorded in the drone logs is saved as a .csv in the input format used by Pix4D.
    - We used Pix4D with an educational license for SfM processing. To follow this notebook without Pix4D, find the generated outputs [here](data repo). For high quality geo-referencing, ground control points should be incorporated at this point in the process with your chosen SfM software.
    - We then [calculate](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/extract_drone_movement.ipynb) how the drone (camera) moves between anchor frames. We can then optionally [confirm](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/intersegment-differences.ipynb) that the local movement estimation was accurate. Combining this local drone movement with the SfM outputs we [project the detected animal tracks into the 3D landscape](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/drone_to_landscape.ipynb).
    - We can [visualize](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/mapping/visualize_tracks_figure.ipynb) the georeferenced tracks on the 3D landscapes.

- **Step 5: Body-part Keypoint Detection**
    - We [extract square crops](https://github.com/benkoger/overhead-video-worked-examples/blob/main/ungulates/postures/crop_out_tracks.ipynb) around each tracked individual. These crops can be used with one of many existing open source animal keypoint tracking softwares such as [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut), [SLEAP](https://sleap.ai/), or [DeepPoseKit](https://github.com/jgraving/DeepPoseKit).
    - We [provide](data repo) complete keypoints for the observation previously generated by DeepPoseKit ([performance stats](https://elifesciences.org/articles/47994)) as well as a [page]() describing how to use DeepLabCut to train a new model to use for keypoint detection in the context of this method.

- **Step 6: Landscape Quantification**
    -

To work through this example in sequence, download the data (insert link to dataset) and start [here](./ungulates/detection/model-training/train_ungulate_detection.ipynb).
