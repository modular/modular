# PyTorch and ONNX inference in MAX

MAX can accelerate the inference of existing PyTorch or ONNX models. These
examples show several common PyTorch or ONNX models running in MAX through the
Mojo, Python, and C APIs:

## PyTorch (via TorchScript)

- BERT, [with the Mojo API](./bert-mojo-torchscript/),
[with the Python API](./bert-python-torchscript/),
and [with the C API](./bert-c-torchscript/)
- [ResNet50 with the Python API](./resnet50-python-torchscript/)

## ONNX

- Stable Diffusion, [with the Mojo API](./stable-diffusion-mojo-onnx/) and
[with the Python API](./stable-diffusion-python-onnx/)
- [YOLOv8 with the Python API](./yolo-python-onnx/)
