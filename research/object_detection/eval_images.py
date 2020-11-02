import random
import matplotlib
import matplotlib.pyplot as plt
import os
import glob

import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont


import tensorflow as tf

from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

category_index = {1: {'id': 1, 'name': 'cima'}, 2: {'id': 2, 'name': 'frente'}}

# recover our saved model
# pipeline_config = '/home/randolfo/models/export/onebox/2etf12/pipeline.config'
# pipeline_config = 'object_detection/data/ssd_efficientdet_d3_896x896_coco17_tpu-32_onebox.config'
# generally you want to put the last ckpt from training in here
# model_dir = '/home/randolfo/models/export/onebox/2etf12/checkpoint'
model_dir = '/home/randolfo/models/export/onebox/2e04_rcnn101_3000/saved_model'
# model_dir = '/home/randolfo/models/training/onebox/2etf12'


# detection_model = tf.saved_model.load(model_dir)
# detect_fn = detection_model.signatures["serving_default"]
detect_fn = tf.saved_model.load(model_dir)

# print(list(detection_model.signatures.keys()))

# configs = config_util.get_configs_from_pipeline_file(pipeline_config)
# model_config = configs['model']
# detection_model = model_builder.build(
#     model_config=model_config, is_training=False)

# # Restore checkpoint
# ckpt = tf.compat.v2.train.Checkpoint(
#       model=detection_model)
# ckpt.restore(os.path.join(model_dir, 'ckpt-11'))


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


# detect_fn = get_model_detection_function(detection_model)


TEST_IMAGE_PATHS = glob.glob('/home/randolfo/images/onebox/samples/*.jpg')

for image_path in TEST_IMAGE_PATHS:
    print(image_path)
    # image_np = load_image_into_numpy_array(image_path)
    image_np = np.array(Image.open(image_path))

    image_np_with_detections = image_np.copy()

    # input_tensor = tf.convert_to_tensor(
    #     np.expand_dims(image_np, 0), dtype=tf.float32)

    # input_tensor = input_tensor / 255
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0))
    detections = detect_fn(input_tensor)

    label_id_offset = 0

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.5,
        agnostic_mode=False,
    )

    dest_dir = '/home/randolfo/images/onebox/eval/eval_2e04_rcnn101_3000'
    os.makedirs(dest_dir, exist_ok=True)
    im = Image.fromarray(image_np_with_detections)
    im.save(os.path.join(dest_dir, os.path.basename(image_path)))
