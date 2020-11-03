import os
import glob
import numpy as np
import tensorflow as tf
import argparse
import pickle

from PIL import Image

from object_detection.utils import visualization_utils as viz_utils


def eval_images(model_dir, samples_dir, dest_dir, out_filename):
    category_index = {1: {"id": 1, "name": "cima"}, 2: {"id": 2, "name": "frente"}}

    tf.get_logger().setLevel("ERROR")

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # detection_model = tf.saved_model.load(model_dir)
    # detect_fn = detection_model.signatures["serving_default"]
    detect_fn = tf.saved_model.load(model_dir)

    image_paths = glob.glob(os.path.join(samples_dir, "*.jpg"))

    dict_result = dict()
    dictionary_output = dict()

    for image_path in image_paths:
        print(image_path)
        image_np = np.array(Image.open(image_path))

        image_np_with_detections = image_np.copy()

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0))
        detections = detect_fn(input_tensor)

        boxes = detections["detection_boxes"][0].numpy()
        classes = detections["detection_classes"][0].numpy()
        scores = detections["detection_scores"][0].numpy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxes,
            classes.astype(int),
            scores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.5,
            agnostic_mode=False,
            line_thickness=1
        )

        file_name = os.path.basename(image_path)

        os.makedirs(dest_dir, exist_ok=True)
        im = Image.fromarray(image_np_with_detections)
        im.save(os.path.join(dest_dir, file_name))

        dict_result["boxes"] = boxes
        dict_result["classes"] = classes
        dict_result["scores"] = scores

        dictionary_output[file_name] = dict_result.copy()

    if out_filename != '':
        file_path_output = os.path.join(dest_dir, out_filename)
        os.makedirs(os.path.dirname(file_path_output),exist_ok=True)
                
        with open(file_path_output, 'wb') as fp:
            pickle.dump(dictionary_output, fp)

def main():
    print(tf.__version__)

    parser = argparse.ArgumentParser(description="Predict objects on samples")
    parser.add_argument("--model", help="Path to model", type=str)
    parser.add_argument("--samples", help="Path to samples", type=str, default="")
    parser.add_argument("--dest", help="Destination path", type=str, default="")
    parser.add_argument("--filename", help="Destination filename to save data", type=str, default="")

    args = parser.parse_args()

    eval_images(args.model, args.samples, args.dest, args.filename)


if __name__ == "__main__":
    main()
