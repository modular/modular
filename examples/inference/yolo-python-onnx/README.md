# YOLOv8 Segmentation with Python

This directory illustrates how to run YOLOv8 Segmentation through the MAX Engine.
Specifically, this example uses Ultralytics YOLO with opencv to segment images
from the webcam.

## Quickstart

This example will change depending on if you have a webcam or not.
If you have a webcam, the example will capture the webcam
and display a window running live segmentation.

If you do not have a webcam, the example will segment
[a downloaded video file](https://drive.google.com/file/d/1H9abV76VohmT-J2RmDrbDhF-FCHt1Sbh/view?usp=sharing)
and generate `output.mp4`.

### Magic instructions

If you have [`magic`](https://docs.modular.com/magic), you can run the
following command:

```sh
magic run bash run.sh
```

### Conda instructions

Create a Conda environment, activate that environment, and install the
requirements:

```sh
# Create a Conda environment if you don't have one
conda create -n max-repo
# Update the environment with the environment.yml file
conda env update -n max-repo -f environment.yml --prune
# Run the example
conda run -n max-repo --live-stream bash run.sh
```

## Note: GUI Dependencies

The webcam version of this example depends on `opencv-python`
and its ability to render GUIs.
The dependencies for this are not always installed on linux.

Downloading these dependencies is distro dependent.
On Ubuntu, it should be:

```bash
apt install -y libgl1
```

## Performance Comparison

To get an idea of the performance difference between the ONNX Runtime and MAX Engine,
you can run the webcam script with/without the `--onnx` flag.

The model execution time is printed as part of the window title.
Note, it can take a bit for the execution time to stabilize.

MAX Engine:

```sh
./segment.py webcam
```

ONNX Runtime:

```sh
./segment.py webcam --onnx
```

## Files

- `download-model.py`: Downloads YOLOv8n-seg from
[ultralytics](https://github.com/ultralytics/ultralytics)
and exports it as ONNX.

- `segment.py`: Example program that runs the full YOLO segmentation pipeline
through the MAX Engine on images from the webcam or video files.
Use `--help` to see the various commands.
