'''
@author: Prathmesh R Madhu.
For educational purposes only
'''

# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import os
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import felzenszwalb

from selective_search import selective_search

def main():
    # Define the relative path to the images
    base_path = '../data'

    # Loop through folders and images
    folders = ['chrisarch', 'arthist', 'classarch']
    for folder in folders:
        folder_path = os.path.join(base_path, folder)

        # Process each image in the folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            print(f"Processing image: {image_path}")

            # Load the image
            image = skimage.io.imread(image_path)
            print(f"Image shape: {image.shape}")

            # Perform selective search
            image_label, regions = selective_search(
                image,
                scale=500,
                min_size=20
            )

            candidates = set()
            for r in regions:
                # Exclude duplicate rectangles
                if r['rect'] in candidates:
                    continue

                # Exclude small regions
                if r['size'] < 2000:
                    continue

                # Exclude distorted rectangles
                x, y, w, h = r['rect']
                if w / h > 1.2 or h / w > 1.2:
                    continue

                candidates.add(r['rect'])

            # Draw rectangles on the original image
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax.imshow(image)
            for x, y, w, h in candidates:
                print(x, y, w, h, r['size'])
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=1
                )
                ax.add_patch(rect)
            plt.axis('off')

            # Save the image to the results folder
            results_path = '../results'
            if not os.path.isdir(results_path):
                os.makedirs(results_path)
            output_path = os.path.join(results_path, f"{folder}_{image_name}")
            fig.savefig(output_path)
            print(f"Saved processed image to: {output_path}")
            plt.close(fig)


if __name__ == '__main__':
    main()

#Original

#
#
# def main():
#
#     # loading a test image from '../data' folder
#     image_path = 'path/to/image.jpg'
#     image = skimage.io.imread(image_path)
#     print (image.shape)
#
#     # perform selective search
#     image_label, regions = selective_search(
#                             image,
#                             scale=500,
#                             min_size=20
#                         )
#
#     candidates = set()
#     for r in regions:
#         # excluding same rectangle (with different segments)
#         if r['rect'] in candidates:
#             continue
#
#         # excluding regions smaller than 2000 pixels
#         # you can experiment using different values for the same
#         if r['size'] < 2000:
#             continue
#
#         # excluding distorted rects
#         x, y, w, h = r['rect']
#         if w/h > 1.2 or h/w > 1.2:
#             continue
#
#         candidates.add(r['rect'])
#
#     # Draw rectangles on the original image
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#     ax.imshow(image)
#     for x, y, w, h in candidates:
#         print (x, y, w, h, r['size'])
#         rect = mpatches.Rectangle(
#             (x, y), w, h, fill=False, edgecolor='red', linewidth=1
#         )
#         ax.add_patch(rect)
#     plt.axis('off')
#     # saving the image
#     if not os.path.isdir('results/'):
#         os.makedirs('results/')
#     fig.savefig('results/'+image_path.split('/')[-1])
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()