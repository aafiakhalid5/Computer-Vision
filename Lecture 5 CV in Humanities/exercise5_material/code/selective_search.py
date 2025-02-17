'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
from skimage.segmentation import felzenszwalb
import numpy as np

def generate_segments(im_orig, scale=300, sigma=0.8, min_size=50):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###
    segments = felzenszwalb(im_orig, scale=scale, sigma=sigma, min_size=min_size)
    print(f"Generated segments with shape: {segments.shape}, unique regions: {len(np.unique(segments))}")

    # Add the segmentation mask as a 4th channel
    im_with_segments = np.dstack((im_orig, segments))

    return im_with_segments


def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###
    return np.sum(np.minimum(r1['color_hist'], r2['color_hist']))


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###

    return np.sum(np.minimum(r1['texture_hist'], r2['texture_hist']))



def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###

    return 1 - (r1['size'] + r2['size']) / imsize


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###
    bbox = [
        min(r1['bbox'][0], r2['bbox'][0]),
        min(r1['bbox'][1], r2['bbox'][1]),
        max(r1['bbox'][2], r2['bbox'][2]),
        max(r1['bbox'][3], r2['bbox'][3])
    ]
    bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return 1 - (bbox_size - r1['size'] - r2['size']) / imsize


def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    ### YOUR CODE HERE ###

    # Convert the image to HSV color space
    hsv_img = skimage.color.rgb2hsv(img)

    # Calculate histograms for each channel
    for channel in range(hsv_img.shape[2]):  # Iterate over HSV channels
        channel_hist, _ = np.histogram(
            hsv_img[:, :, channel], bins=BINS, range=(0, 1)
        )
        # Normalize histogram using L1 norm
        channel_hist = channel_hist / channel_hist.sum()
        hist = np.concatenate((hist, channel_hist))  # Append to the final histogram

    return hist

def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###

    # Parameters for LBP
    radius = 1
    n_points = 8 * radius

    # Compute LBP for each channel
    for channel in range(img.shape[2]):  # Iterate over color channels
        ret[:, :, channel] = local_binary_pattern(
            img[:, :, channel], n_points, radius, method='uniform'
        )
    return ret

def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###

    # Calculate the gradient using calc_texture_gradient
    gradient = calc_texture_gradient(img)

    # Calculate histograms for each channel of the gradient
    for channel in range(gradient.shape[2]):  # Iterate over gradient channels
        channel_hist, _ = np.histogram(
            gradient[:, :, channel], bins=BINS, range=(0, np.max(gradient))
        )
        # Normalize the histogram using L1 norm
        channel_hist = channel_hist / channel_hist.sum()
        hist = np.concatenate((hist, channel_hist))  # Append to the final histogram

    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    ### YOUR CODE HERE ###

    imsize = img.shape[0] * img.shape[1]

    # Calculate the texture gradient
    gradient = calc_texture_gradient(img)

    for region_label in np.unique(img[:, :, 3]):  # Loop through unique regions
        mask = img[:, :, 3] == region_label  # Binary mask for this region

        # Extract individual region
        region_pixels = img[:, :, :3][mask]  # Original image pixels for this region
        region_gradient = gradient[mask]  # Gradient pixels for this region

        # Store region properties
        R[region_label] = {
            'size': np.sum(mask),  # Total number of pixels
            'color_hist': calc_colour_hist(region_pixels),  # Color histogram
            'texture_hist': calc_texture_hist(region_gradient),  # Texture histogram
            'bbox': [
                np.min(np.where(mask)[1]),  # Min X
                np.min(np.where(mask)[0]),  # Min Y
                np.max(np.where(mask)[1]),  # Max X
                np.max(np.where(mask)[0])  # Max Y
            ]
        }

    return R

def extract_neighbours(regions):
    """
    Extract a list of neighboring regions.
    :param regions: A dictionary containing region properties.
    :return: A list of neighboring region pairs.
    """

    def intersect(a, b):
        """
        Check if two bounding boxes intersect.
        :param a: First region's bounding box [min_x, min_y, max_x, max_y].
        :param b: Second region's bounding box [min_x, min_y, max_x, max_y].
        :return: True if the regions intersect, False otherwise.
        """
        return not (
            a["max_x"] < b["min_x"] or
            a["min_x"] > b["max_x"] or
            a["max_y"] < b["min_y"] or
            a["min_y"] > b["max_y"]
        )

    # Initialize an empty list to store neighbor pairs
    neighbours = []
    ### YOUR CODE HERE ###
    
    # Convert the regions dictionary into a list of tuples for easier iteration
    region_list = list(regions.items())

    # Iterate through all pairs of regions
    for i, (label_a, region_a) in enumerate(region_list):
        for label_b, region_b in region_list[i + 1:]:
            # Check if the regions are neighbors using the intersect function
            if intersect(region_a['bbox'], region_b['bbox']):
                neighbours.append((label_a, label_b))

    return neighbours


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE

    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###


        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###


        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###


    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###


    return image, regions


