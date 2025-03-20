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
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from max import engine
from shared import menu, modular_cache_dir
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import ops

st.set_page_config("YOLO", page_icon="🔍")
menu()

"""
# 🔍 YOLO Segmentation

Segment objects using your webcam. This downloads and converts YOLOv8n to
ONNX, then compiles it with MAX for faster inference!
"""

# First do a quick check to see if a webcam is available
capture = cv2.VideoCapture(cv2.CAP_ANY)
if capture.isOpened():
    capture.release()
else:
    st.error("This example is only available on a local machine with a webcam")
    exit()


@st.cache_data(show_spinner="Downloading YOLO and exporting to ONNX")
def download_and_export_yolo(model_path, height, width):
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=(height, width), simplify=True)
    return model.names


@st.cache_resource(show_spinner="Starting MAX Inference Session")
def max_yolo_session(onnx_path):
    # Have to sleep for a short time for the spinner to start correctly
    time.sleep(1)
    session = engine.InferenceSession()
    return session.load(onnx_path)


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


def postprocess(out0, out1, input, frame, class_names):
    out0 = torch.from_numpy(out0)
    out1 = torch.from_numpy(out1)

    pred = ops.non_max_suppression(
        out0,
        conf_thres=0.25,
        iou_thres=0.70,
        agnostic=False,
        max_det=10,
        nc=len(class_names),
        classes=None,
    )[0]

    if not len(pred):
        result = Results(
            orig_img=frame,
            path="",
            names=class_names,
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
            names=class_names,
            boxes=pred[:, :6],
            masks=masks,
        )
    return result


model_path = st.sidebar.text_input(
    "Model Path",
    os.path.join(modular_cache_dir(), "yolov8n-seg.pt"),
)
onnx_path = Path(os.path.dirname(model_path)) / "yolov8n-seg.onnx"
width = st.sidebar.number_input("Image Width", 64, 2048, 640)
height = st.sidebar.number_input("Image Height", 64, 2048, 480)

class_names = download_and_export_yolo(model_path, height, width)
yolo = max_yolo_session(onnx_path)

previous_elapsed_ms = []

frame_window = st.image([])
camera = cv2.VideoCapture(0)

button_placeholder = st.empty()
if button_placeholder.button("Start Webcam"):
    button_placeholder.empty()
    while True:
        _, img = camera.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_and_pad(img, (height, width))
        # Preprocess inputs.
        input = (
            img[np.newaxis, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32)
            / 255
        ).copy()
        start = time.time()
        outputs = list(yolo.execute_legacy(images=input).values())
        elapsed_ms = (time.time() - start) * 1000
        result = postprocess(outputs[0], outputs[1], input, img, class_names)
        img = result.plot()
        # Calculated average fps and update window title.
        # global previous_elapsed_ms
        previous_elapsed_ms.append(elapsed_ms)
        previous_elapsed_ms = previous_elapsed_ms[-100:]
        fps = 1000.0 / np.average(previous_elapsed_ms)
        img = cv2.putText(
            img,
            f"FPS: {int(fps)}",
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
        )
        frame_window.image(img)
