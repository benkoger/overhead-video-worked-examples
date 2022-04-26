import os
import cv2

def save_frame(cap, frame_num, outfile, crop_size=None, top_left=None):
    """ Save frame from cv2 VideoCapture
    
    Args: 
        cap: cv2.VideoCapture object
        frame_num: the frame number to save
        outfile: where to save frame
        crop_size: pixels, size of crop (square). If None, no crop.
        top_left: (i, j) coordinate of top left corner of crop (if not None)
            if None and crop_size is not None, then choose random values
            
    Return crop_top_left
        
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if frame is not None:
        if crop_size:
            if not top_left:
                raise ValueError(f"If cropping, must provide top_left: {top_left}")
            top, left = top_left
            frame = frame[top:top+crop_size, left:left+crop_size]

        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        cv2.imwrite(outfile, frame)
    else:
        print(f"with frame to be saved at outfile {outfile}.")