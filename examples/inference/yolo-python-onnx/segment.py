#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
import os
import platform
import shutil
import signal
import subprocess
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import onnxruntime
import torch
from constants import CLASS_NAMES
from max.engine import InferenceSession
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import ops

DESCRIPTION = "Segment images from a webcam or video file using YOLO."
DEFAULT_MODEL_DIR = "../../models/yolo"
DEFAULT_INPUT_FILE = "input.mp4"
DEFAULT_OUTPUT_FILE = "output.mp4"
WINDOW_NAME = "YOLOv8 Segmentation"


def resize_and_pad(image, shape):
    # Grab shape sizes.
    (h, w, _) = image.shape
    (target_h, target_w) = shape

    # Resize to fully fit within `shape`.
    min_ratio = min(target_h / h, target_w / w)
    unpadded_h = int(round(min_ratio * h))
    unpadded_w = int(round(min_ratio * w))
    image = cv2.resize(
        image, (unpadded_w, unpadded_h), interpolation=cv2.INTER_LINEAR
    )

    # Pad to be the same size as `shape`.
    delta_h = (target_h - unpadded_h) / 2
    delta_w = (target_w - unpadded_w) / 2
    top, bottom = int(round(delta_h - 0.1)), int(round(delta_h + 0.1))
    left, right = int(round(delta_w - 0.1)), int(round(delta_w + 0.1))
    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def postprocess(out0, out1, input, frame):
    out0 = torch.from_numpy(out0)
    out1 = torch.from_numpy(out1)

    pred = ops.non_max_suppression(
        out0,
        conf_thres=0.25,
        iou_thres=0.70,
        agnostic=False,
        max_det=10,
        nc=len(CLASS_NAMES),
        classes=None,
    )[0]

    if not len(pred):
        result = Results(
            orig_img=frame,
            path="",
            names=CLASS_NAMES,
            boxes=pred[:, :6],
        )
    else:
        masks = ops.process_mask(
            out1[0],
            pred[:, 6:],
            pred[:, :4],
            input.shape[2:],
            upsample=True,
        )  # HWC
        pred[:, :4] = ops.scale_boxes(input.shape[2:], pred[:, :4], frame.shape)
        result = Results(
            orig_img=frame,
            path="",
            names=CLASS_NAMES,
            boxes=pred[:, :6],
            masks=masks,
        )
    return result


def camera_functional():
    cap = cv2.VideoCapture(0)
    if cap is None:
        return False
    if not cap.isOpened():
        cap.release()
        return False

    can_read_frame, _ = cap.read()
    cap.release()
    return can_read_frame


def process_webcam(args):
    # Compile & load models - this may take a few minutes.
    print("Loading and compiling model...")
    if args.onnx:
        engine_name = "ONNX Runtime"
        yolo = onnxruntime.InferenceSession(
            f"{args.model_dir}/yolov8n-seg.onnx"
        )
    else:
        engine_name = "MAX Engine"
        session = InferenceSession()
        yolo = session.load(f"{args.model_dir}/yolov8n-seg.onnx")
    print("Model compiled.\n")

    # Setup camera.
    print("Grabbing camera input...")
    print("Press escape or q to quit")
    cap = cv2.VideoCapture(0)

    if cap is None or not cap.isOpened():
        print("Failed to open stream from webcam.")
        print(
            "If you don't have a webcam, try running with the `video`"
            " subcommand instead."
        )
        if cap is not None:
            cap.release()
        exit(1)

    previous_elapsed_ms = []

    # Loop camera frames running yolo.
    while cap.isOpened():
        # Load frame from camera.
        success, frame = cap.read()

        if not success:
            print("Can't load frame. Camera stream ended?")
            break

        # Resize to 640x480
        frame = resize_and_pad(frame, (480, 640))

        # Preprocess inputs.
        input = (
            frame[np.newaxis, :, :, ::-1]
            .transpose(0, 3, 1, 2)
            .astype(np.float32)
            / 255
        ).copy()

        # Run prediction.
        start = time.time()
        if args.onnx:
            outputs = yolo.run(None, {"images": input})
        else:
            outputs = list(yolo.execute(images=input).values())
        elapsed_ms = (time.time() - start) * 1000

        # Postprocess outputs.
        result = postprocess(outputs[0], outputs[1], input, frame)

        # Annotate and display frame.
        annotated_frame = result.plot()
        try:
            cv2.imshow(WINDOW_NAME, annotated_frame)
        except:
            print("Failed to open window to display the annotated frame.")
            print(
                "If you don't have a gui, try running with the `video`"
                " subcommand instead."
            )
            break

        # Calculated average fps and update window title.
        previous_elapsed_ms.append(elapsed_ms)
        previous_elapsed_ms = previous_elapsed_ms[-100:]
        average_elapsed_ms = np.average(previous_elapsed_ms)
        cv2.setWindowTitle(
            WINDOW_NAME,
            (
                f"{engine_name}: YOLOv8 Segmentation (Average Model Time:"
                f" {average_elapsed_ms:0.1f}ms)"
            ),
        )

        # Exit on escape or q.
        ESC = 27
        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ESC]:
            break

    # Clean up resources.
    cap.release()
    cv2.destroyAllWindows()


def process_video(args):
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        print("You can use your own video file with the --input flag")
        exit(1)

    # Compile & load models - this may take a few minutes.
    print("Loading and compiling model...")
    session = InferenceSession()
    yolo = session.load(f"{args.model_dir}/yolov8n-seg.onnx")
    print("Model compiled.\n")

    # Setup video capture.
    print("Processing input video...")
    cap = cv2.VideoCapture(args.input)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(args.output, fourcc, fps, (640, 480))

    # Loop camera frames running yolo.
    while cap.isOpened():
        # Load frame from camera.
        success, frame = cap.read()

        if not success:
            # Out of frames. Everything is done.
            break

        # Resize to 640x480
        frame = resize_and_pad(frame, (480, 640))

        # Preprocess inputs.
        input = (
            frame[np.newaxis, :, :, ::-1]
            .transpose(0, 3, 1, 2)
            .astype(np.float32)
            / 255
        ).copy()

        # Run prediction.
        outputs = list(yolo.execute(images=input).values())

        # Postprocess outputs.
        result = postprocess(outputs[0], outputs[1], input, frame)

        # Annotate and output frame.
        annotated_frame = result.plot()
        out.write(annotated_frame)

    # Clean up resources.
    cap.release()
    out.release()

    # Full path so users can find video easily
    output_name = f"{os.getcwd()}/{args.output}"

    # Workaround for linux python-opencv/ffmpeg not being able to encode h264
    if shutil.which("ffmpeg"):
        print("Changing video encoding for wider video player support.")
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                output_name,
                "-vcodec",
                "libx264",
                "encoded.mp4",
            ]
        )

        # If succesfully encoded write over original file
        if result.returncode == 0:
            subprocess.run(["mv", "encoded.mp4", output_name])
        else:
            print("\nFailed to encode video, but video may still be playable.")

    # Open the video if running in vscode and `code` is on path
    if "TERM_PROGRAM" in os.environ and os.environ["TERM_PROGRAM"] == "vscode":
        if shutil.which("code"):
            result = subprocess.run(["code", output_name])

    print("Video saved to:", output_name)


def main():
    # Parse args.
    parser = ArgumentParser(description=DESCRIPTION)
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    webcam_parser = subparsers.add_parser(
        "webcam", help="Segment images from the webcam and display them"
    )
    webcam_parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the downloaded model",
    )
    webcam_parser.add_argument(
        "--onnx",
        "--onnx-runtime",
        action="store_true",
        default=False,
        help=(
            "Run video segmentation with the ONNX Runtime to compare"
            " performance"
        ),
    )
    video_parser = subparsers.add_parser("video", help="Segment a video file")
    video_parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the downloaded model",
    )
    video_parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Input video file to segment",
    )
    video_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output video file to write to",
    )
    args = parser.parse_args()

    # Improves model compilation speed dramatically on intel CPUs
    if "Intel" in platform.processor():
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if args.command == "webcam":
        process_webcam(args)
    elif args.command == "video":
        process_video(args)
    else:
        # No command was given. First attempt to run webcam with default value.
        # If there is no webcam, run video segmentation with default values.
        args.model_dir = DEFAULT_MODEL_DIR
        args.input = DEFAULT_INPUT_FILE
        args.output = DEFAULT_OUTPUT_FILE
        args.onnx = False

        # Check if a webcam is available.
        if camera_functional():
            print("Webcam is available.")
            print("Running live segmentation.\n")
            process_webcam(args)
        else:
            print("Webcam was not available.")
            print("Falling back to video file segmentation.\n")
            process_video(args)


if __name__ == "__main__":
    main()
