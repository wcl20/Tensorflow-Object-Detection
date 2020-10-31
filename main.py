import argparse
import cv2
import numpy as np
from core.utils.detection import classify_rois
from core.utils.detection import image_pyramid
from core.utils.detection import sliding_window
from imutils.object_detection import non_max_suppression
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--confidence", type=float, default=0.5, help="confidence level to filter weak predictions")
    args = parser.parse_args()

    # Input size of model / ROI
    input_size = (224, 224)

    model = ResNet50(weights="imagenet", include_top=True)

    image = cv2.imread(args.image)
    height, width = image.shape[:2]
    image_size = (350, 350)
    resized = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)

    labels = {}
    rois = []; locations = []
    for scaled in image_pyramid(resized, scale=1.5, min_size=input_size):
        for x, y, roi in sliding_window(scaled, step=16, window_size=input_size):
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            roi = imagenet_utils.preprocess_input(roi)
            rois.append(roi)
            locations.append((x, y))

    rois = np.vstack(rois)
    labels = classify_rois(model, rois, locations, labels, confidence=args.confidence)
    # Perform NMS for each class
    for label in labels.keys():
        clone = resized.copy()
        bboxs = np.array([predictions[0] for predictions in labels[label]])
        probs = np.array([predictions[1] for predictions in labels[label]])
        # Without NMS
        for x1, y1, x2, y2 in bboxs:
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # NMS
        bboxs = non_max_suppression(bboxs, probs)
        for x1, y1, x2, y2 in bboxs:
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(f"{label}.png", clone)


if __name__ == '__main__':
    main()
