import imutils
from tensorflow.keras.applications import imagenet_utils

def sliding_window(image, step, window_size):
    height, width = image.shape[:2]
    window_width, window_height = window_size
    for y in range(0, height - window_height, step):
        for x in range(0, width - window_width, step):
            yield (x, y, image[y:y+window_height, x:x+window_width])

def image_pyramid(image, scale=1.5, min_size=(224, 224)):
    # Return original image
    yield image
    while True:
        # Downsize image
        new_width = int(image.shape[1] / scale)
        image = imutils.resize(image, width=new_width)
        # Break if image is smaller than minsize
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image

def classify_rois(model, rois, locations, labels, confidence=0.5, top=10, dims=(224, 224)):

    # Get top K predictions from model
    preds = model.predict(rois, batch_size=64)
    preds = imagenet_utils.decode_predictions(preds, top=top)

    for i, pred in enumerate(preds):
        for _, label, prob in pred:
            if prob > confidence:
                x, y = locations[i]
                bbox = (x, y, x + dims[0], y + dims[1])
                # Get prediction list for the label and append prediciton
                label_list = labels.get(label, [])
                label_list.append((bbox, prob))
                labels[label] = label_list
    return labels
