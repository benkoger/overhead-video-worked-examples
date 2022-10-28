import numpy as np

def random_top_left(im_shape, crop_size, gaussian=False):
    """ Get a random top left coordinate for a crop of size (crop_size * crop_size).
    
    Args:
        im_shape: (h, w, ...)
        crop_size: size of ultimate crop
        gaussian: If True, then pull coordinates from gaussian with mean
            at the center of possible range of top left values and 1 standard 
            deviation to the min and max top left values
    
    Returns [top, left]
    """
    
    height, width = im_shape[:2]
    if gaussian:
        mean_top = (width-crop_size) / 2
        mean_left = (width-crop_size) / 2
        top = -1
        left = -1
        while ((top >= (height-crop_size)) or (top < 0)):
            top = int(np.random.normal(mean_top, mean_top))
        while ((left >= (width-crop_size)) or (left < 0)):
            left = int(np.random.normal(mean_left, mean_left))
    else:
        top = np.random.randint(0, height-crop_size)
        left = np.random.randint(0, width-crop_size)
    top_left = [top, left]
    return top_left