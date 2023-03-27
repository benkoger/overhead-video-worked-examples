### Within the geladas folder there is a JSON file called local-paths.json that the various notebooks use to load and save data locally. These paths will be unique to the file path structure of the computer being used. Some keys may point to the same local path as the user chooses. The following fields are expected:

**annotations_folder**: This is the path to a folder containing images and corresponding annotations. This folder contains coco format .json files that contain the annotation information for the images being used. It is expected there is at least a .json file for training and validation and likely for testing. There could also be more various subsets of the annotated data. There is also a folder called ‘annotated_images,’ contained within this folder that holds the annotated images. This structure allows images to be reused in multiple annotation collections without needing to be saved in many locations.
The expected folder structure is:
  - annotations_folder:
    - train.json
    - val.json
    - …
    - annotated_images
      - image_file0.jpg
      - image_file1.jpg
      - ...

**base_gelada**: Path to folder that will be used to store various generated data such as detections and tracks. A folder will be created for each observation in this folder as part the processing notebooks.

**detectron_path**: This is the path to where detectron2 is installed locally on your computer. It is used to access pretrained weights for various networks. 

**figure_folder**: Path to folder where figures will be saved.

**labelbox_key**: This isn't a path. It is the labelbox api key for the labelbox account being used. Can be found on the labelbox website in the users account area. 

**overhead_functions_path**: This is the path to the folder called “functions” in this repository

**pix4d_folder**: Path to folder that will store the various structure from motion processing inputs and outputs (in our case pix4d). Within this folder each observation has its own folder created in the geladas/mapping/get_anchor_frames.ipynb notebook.

**videos_folder**: Path to folder that contains the observation video files






