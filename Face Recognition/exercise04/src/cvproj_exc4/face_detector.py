import cv2
import numpy as np
from mtcnn import MTCNN


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=25, tm_threshold=0.2, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # TODO: Specify all parameters for template matching.
        self.tm_threshold = tm_threshold
        self.tm_window_size = tm_window_size



    # TODO: Track a face in a new image using template matching.
    def track_face(self, image):
        if self.reference is None:
            # Detect face in the initial frame and set it as reference.
            detection = self.detect_face(image)
            if detection is not None:
                self.reference = detection['rect']
                return detection
            else:
                return None

        # Define the region of interest (ROI) for template matching.
        x, y, w, h = self.reference
        search_window = [
            max(0, x - self.tm_window_size),
            max(0, y - self.tm_window_size),
            min(image.shape[1], x + w + self.tm_window_size),
            min(image.shape[0], y + h + self.tm_window_size),
        ]

        # Extract the template and search area from the image.
        template = self.crop_face(image, self.reference)
        search_area = image[
                      search_window[1]:search_window[3],
                      search_window[0]:search_window[2],
                      ]

        # Perform template matching.
        result = cv2.matchTemplate(search_area, template, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)

        if min_val > self.tm_threshold:
            # Re-initialize tracking if similarity is below threshold.
            detection = self.detect_face(image)
            if detection is not None:
                self.reference = detection['rect']
                return detection
            else:
                self.reference = None
                return None

        # Update reference based on tracking result.
        new_x, new_y = (
            search_window[0] + min_loc[0],
            search_window[1] + min_loc[1],
        )
        self.reference = [new_x, new_y, w, h]

        # Align and return the tracked face.
        aligned = self.align_face(image, self.reference)
        return {"rect": self.reference, "image": image, "aligned": aligned, "response": min_val}


    # def track_face(self, image):
    #     # If no reference exists, detect a face and set it as the reference.
    #     if self.reference is None:
    #         detection = self.detect_face(image)
    #         if detection is None:  # No face detected
    #             return None
    #         self.reference = detection
    #         return detection
    #
    #     # Define the search window around the previous face position
    #     ref_rect = self.reference["rect"]
    #     top = max(ref_rect[1] - 25, 0)
    #     left = max(ref_rect[0] - 25, 0)
    #     bottom = min(ref_rect[1] + ref_rect[3] + 25, image.shape[0])
    #     right = min(ref_rect[0] + ref_rect[2] + 25, image.shape[1])
    #
    #     # Extract the search window from the image
    #     search_window = image[top:bottom, left:right]
    #
    #     # Extract the template from the reference
    #     template = self.crop_face(self.reference["image"], self.reference["rect"])
    #
    #     # Perform template matching
    #     result = cv2.matchTemplate(search_window, template, cv2.TM_SQDIFF_NORMED)
    #     min_val, _, min_loc, _ = cv2.minMaxLoc(result)
    #
    #     # Check if the similarity score is above the threshold
    #     if min_val > self.tm_threshold:
    #         # Tracking failed; reinitialize with detect_face
    #         detection = self.detect_face(image)
    #         if detection is None:
    #             self.reference = None  # No face detected
    #             return None
    #         self.reference = detection
    #         return detection
    #
    #     # Update reference with the new detected position
    #     new_top_left = (left + min_loc[0], top + min_loc[1])
    #     new_rect = [
    #         new_top_left[0],
    #         new_top_left[1],
    #         template.shape[1],
    #         template.shape[0],
    #     ]
    #     aligned = self.align_face(image, new_rect)
    #     self.reference = {"rect": new_rect, "image": image, "aligned": aligned, "response": min_val}
    #     return self.reference



    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        if not (detections := self.detector.detect_faces(image)):
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
