import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


PATH_TO_MODEL_DIR = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
PATH_TO_LABELS = 'label_map.pbtxt'
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
class_id = 1

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def get_coordinates(image_path: str) -> np.array:
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    boxes = []
    classes = []
    scores = []
    for i, x in enumerate(detections['detection_classes']):
        if x == class_id and detections['detection_scores'][i] > 0.35:
            classes.append(x)
            boxes.append(detections['detection_boxes'][i])
            scores.append(detections['detection_scores'][i])
    boxes = np.array(boxes)
    classes = np.array(classes)
    scores = np.array(scores)
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          boxes,
          classes,
          scores,
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    return boxes








