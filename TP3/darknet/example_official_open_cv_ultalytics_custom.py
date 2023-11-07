import argparse
import cv2
import numpy as np
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def darken_image(img, alpha):
    # Applique un effet de fondu à l'image
    return cv2.addWeighted(img, 1 - alpha, img, 0, 0)

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """

    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def highlight_detection_area(img, x, y, x_plus_w, y_plus_h, alpha):
    # Crée un masque pour les zones en dehors de la boîte de détection
    mask = np.zeros_like(img)
    mask[y:y_plus_h, x:x_plus_w, :] = 255
    # Applique un effet de fondu sur le masque
    faded_mask = cv2.addWeighted(mask, alpha, mask, 0, 0)
    # Inverse le masque pour éclairer l'intérieur de la boîte
    inverted_mask = cv2.bitwise_not(faded_mask)
    # Fusionne l'image d'origine et le masque inversé
    highlighted_area = cv2.bitwise_and(img, inverted_mask)
    return highlighted_area

def main(onnx_model, input_image, target_classes):
    """
    Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Args:
        onnx_model (str): Path to the ONNX model.
        input_image (str): Path to the input image.

    Returns:
        list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
    """
    # Load the ONNX model
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # Read the input image
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            
            # Vérifier si la classe actuelle fait partie des classes cibles spécifiées en ligne de commande
            if maxScore >= 0.25 and CLASSES[maxClassIndex] in target_classes:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

    faded_image = darken_image(original_image, alpha=0.5)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale}
        detections.append(detection)
        draw_bounding_box(faded_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
        # Applique l'effet d'éclairage à l'intérieur de la boîte de détection
        faded_image = highlight_detection_area(faded_image, round(box[0] * scale), round(box[1] * scale),
                                              round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale), alpha=0.8)

    # Display the image with bounding boxes
    cv2.imshow('image', faded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov5nu.onnx', help='Input your ONNX model.')
    parser.add_argument('--img', default='my_images/img1.jpg', help='Path to input image.')
    parser.add_argument('--classes', nargs='+', default=CLASSES, help='List of target classes to consider')
    args = parser.parse_args()
    main(args.model, args.img, args.classes)