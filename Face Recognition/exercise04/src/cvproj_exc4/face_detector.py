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
                self.template = self.crop_face(image, self.reference)  # Store the initial template
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

        # Extract the search area from the image.
        search_area = image[
                    search_window[1]:search_window[3],
                    search_window[0]:search_window[2],
                    ]

        # Perform template matching.
        result = cv2.matchTemplate(search_area, self.template, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)

        if min_val > self.tm_threshold:
            # Re-initialize tracking if similarity is below threshold.
            detection = self.detect_face(image)
            if detection is not None:
                self.reference = detection['rect']
                self.template = self.crop_face(image, self.reference)  # Update the template
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

        # Update the template with the new face region.
        self.template = self.crop_face(image, self.reference)

        # Align and return the tracked face.
        aligned = self.align_face(image, self.reference)
        return {"rect": self.reference, "image": image, "aligned": aligned, "response": min_val}

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
