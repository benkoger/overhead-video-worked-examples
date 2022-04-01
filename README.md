# overhead-video-worked-examples
Worked examples of behavioral data extraction from overhead video.

Worked Examples:

Kenyan Ungulates

Our dataset had five object classes: zebra, African buffalo, impala, water- buck, and a final class for any species that was annotated just a small number of times we call “other”. The “other” species, like giraffe or eland, shouldn’t be considered background but aren’t worth trying to specifically learn. Our training set was primarily made for detecting zebras. There were 1913 total annotated video frames (the equivalent of annotating just over a minute of video)  with 20,682 zebra instances, 8794 buffalo instances, 6992 impala instances, 800 waterbuck instances, and 1175 “other” instances. The total dataset was randomly distributed into a non overlapping training, validation, and test set where 60 percent of frames were used in the training set and 20 percent used for each the validation and test set.
During training, we randomly flip the training images both vertically and horizontally. We also randomly change the brightness, contrast, and saturation of the images


Training the model:
	Annotating images:
After collecting videos, extract images for annotation
model_training/extract_annotation_images.ipynb
Using the software of your choice, annotate the extracted images
The exact output format depends on the model training pipeline you use. For these examples, the data should be extracted as .json files in the COCO format
Generally, the training and validation set should be separate, but in some cases it is helpful to randomly divide a set of already annotated images into a training and test set

	
Mapping:
All necessary notebooks are found at https://github.com/benkoger/overhead-video-worked-examples/tree/main/ungulates/mapping.
Use the notebook called “link_frames_to_drone_logs.ipynb” to choose the anchor frames with which to build the 3D landscape models. In this example we use the flight logs that DJI drones export as part of a flight (ours are in the format used in 2018). These drone logs don’t explicitly link the frames in the video to the recorded sensor information in this file. While it does record when the camera is recording we didn’t find this to always be reliable. Instead, we manually record the first frame in the video in which the drone visibly begins to return home at the end of an observation.  The flight log records when the “go home” signal is received by the drone from the controller. We link the first major rotation (based on the on board compass) after the “go home” signal has been received to the manually recorded frame. We then estimate the corresponding frame number for every other row of the flight log knowing the time elapsed before or after the “go home” frame number and the fact that we record at 60 frames per second. We also record which video clip (since DJI stores long recordings as smaller clips) the “go home” frame occurs in.
The notebook contains commented threshold variables that define how often a new anchor frame should be selected based on the estimated drone movement and rotation since the last anchor frame.
The selected anchor frames are saved in a specified folder for structure from motion processing with the users chosen software.
Additionally the modified flight logs that record which frames are used for the anchor frames are saved as .pkl files
The estimated latitude, longitude, and elevation coordinates of the drone when each anchor frame was taken is also saved as a .csv the the expected format to be used with pix4D
We use Pix4D (with an educational license) to take the images and corresponding drone location information to build 3D landscape models. …
We use the notebook called “extract_drone_movement.ipynb” to calculate how the drone (camera) moves between anchor frames. Based on output files from Pix4D we note any anchor frames that were removed during the SfM process. Using local features (see the function get_segment_drone_movement in drone_movement.py for specific implementation details) we estimate drone movement from each previous anchor frame.

Geladas
Model training
All notebooks referred to in this section can be found at https://github.com/benkoger/overhead-video-worked-examples/tree/main/geladas/detection/model-training 
We manually split observation videos into training, validation, and test observations with 20 videos used for training and 7 each used  validation and testing. We save this information in a file called video_train_val_test_split.json. This file is used by the notebook called extract_annotation_images.ipynb to extract frames (or crops of frames) from these videos and save them for annotation. In addition to the randomly chosen frames the notebook can also save a frame just before and just after the chosen frame (the spacing is defined by a variable called “triplet_spacing”). These additional views can help during annotation because moving individuals are easier to see across pairs of frames instead of single static images.
For annotation, we use Labelbox (REF:LABELBOX) with a free educational license. We use bounding boxes to annotate all geladas and humans in the frames. Each gelada box is further annotated with a “posture” label (one of: standing, sitting, unknown) and a “status” label (one of: juvenile, adult male, other, unknown). Initially we just annotate a subset of the image sets to train an initial model for model assisted learning (see vi).
After annotating with label box, we use coco_from_labelbox.ipynb to convert the .json file that labelbox exports into the coco .json format that is required by the models we plan to train. We use Detectron2 which is built on top of pytorch to train our model. For this task we use Faster RCNN with a Resnet-50 Feature pyramid network backbone. We use train_gelada_detection.ipynb to train our model.
We train with a simple learning rate scheduler that decreases the learning rate by half after twenty epochs of no improvement on the validation set
We save the model that produces the lowest loss on the validation set over the course of training.
After initial training, we use the model to help annotate further images (Model assisted labeling). Label box has built in functionality for this although, at the time of this work at least, the Labelbox SDK must be used to upload the annotations. We use labelbox_mal.ipynb for this process.
After iterating over step v and vi until appropriate model performance has been achieved, the researcher is left with a robust detection model they can move forward with. The notebook called precision-accuracy-curves.ipynb can be used to evaluate current model performance beyond simply looking at the loss function.

Inference
		All notebooks referred to in this section can be found at https://github.com/benkoger/overhead-video-worked-examples/tree/main/geladas/detection/inference 
The input to the model we just trained is images, not video. While it is very possible to extract frames from video as part of the inference pipeline such that no still frames ever need to be saved (saving substantial storage space), we refer back to the individual video frames at multiple points over the course of our complete method so choose to extract all frames and save them for use both during inference and also later in the process. We use the notebook called extract_video_frames.ipynb to extract all frames.
 The notebook process-video.ipynb uses the previously trained model on all frames across full videos. The user specifies the name of the video that should be processed, and the name of the trained model to use (with the variables ‘video_name’ and ‘model_name’ respectively).
For each frame in the video, the detected bonding boxes (‘pred_boxes’), confidence scores (‘scores’), predicted object classes (‘pred_classes’), and the frame’s name (‘filename’) are saved in a numpy file called [video_name]_detections.npy
Tracking
Mapping
In this example, we demonstrate how to build the maps without corresponding drone logs. This may happen if using old videos or potentially with other brands of drones that save or record from different (or no) sensors compared to the DJI Drones used here. Instead, using “get_anchor_frames.ipynb” we choose anchor frames based on local movement. We need a new anchor frame before too much error accumulates in visual local drone movement estimation. So, we run that exact process but with higher quality thresholds selecting a new anchor frame whenever we dip below the set threshold of number of features used to estimate drone movement. We use an inlier threshold of 100, and 7 pseudo anchor frames between actual anchors.
Once these frames are selected, they are fed into Pix4D to generate 3D models of the landscape and resulting auxiliary files
