"""Testing script."""

import os
from unittest.mock import patch

PATH_ROOT = os.path.dirname(os.path.dirname(__file__))
PATH_DATA = os.path.join(PATH_ROOT, 'data')
PATH_CONFIGS = os.path.join(PATH_ROOT, 'config')
PATH_WEIGHTS = os.path.join(PATH_ROOT, 'weights')


def test_train(tmpdir):
    from scripts.run_train import run_cli

    with patch("argparse._sys.argv", [
        "script.py",
        "--epochs", "2",
        "--batch_size", "1",
        "--model_def", os.path.join(PATH_CONFIGS, "yolov3-tiny.cfg"),
        "--path_output", str(tmpdir),
        "--data_config", os.path.join(PATH_CONFIGS, "custom.data"),
        "--img_size", "416",
        "--augment", "hflip",
        "--grad_accums", "1",
    ]):
        run_cli()


def test_detect(tmpdir):
    from scripts.run_detect import run_cli

    with patch("argparse._sys.argv", [
        "script.py",
        "--image_folder", os.path.join(PATH_DATA, "samples"),
        "--model_def", os.path.join(PATH_CONFIGS, "yolov3-tiny.cfg"),
        "--weights_path", os.path.join(PATH_WEIGHTS, "yolov3-tiny.weights"),
        "--class_path", os.path.join(PATH_DATA, "coco.names"),
        "--output_folder", str(tmpdir),
    ]):
        run_cli()
