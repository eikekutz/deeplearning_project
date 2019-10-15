from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import slic, join_segmentations
from skimage.morphology import watershed
from skimage.color import label2rgb
from skimage import data
import skimage.feature
import numpy as np

def ws(image_pred,th=50.0):
    image = (image_pred[0, 0].numpy()*100).astype(int)
    # Make segmentation using edge-detection and watershed.
    edges = sobel(image)
    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < th] = background
    markers[image >= th] = foreground

    ws = watershed(edges, markers)
    seg1 = label(ws == foreground)

    # return the segmentations.
    return image,seg1
def closest_center(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)