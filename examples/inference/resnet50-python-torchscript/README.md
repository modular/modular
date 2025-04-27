# TorchScript ResNet-50 inference with Python

This directory includes scripts used to run simple ResNet-50 inference via the
MAX Engine Python API to classify an input image. In this case, we use an image
of a leatherback turtle as an example.

## Quickstart

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

## Scripts included

- `download-model.py`: Downloads the model from HuggingFace, converts it to
TorchScript, and saves it to an output directory of your choosing, or defaults
to `../../models/resnet50.torchscript`.

    For more information about the model, please refer to the
    [model card](https://huggingface.co/microsoft/resnet-50).

- `simple-inference.py`: Classifies example input image using MAX Engine.
The script prepares an example input, executes the model, and generates the
resultant classification output.

    You can use the `--input` CLI flag to specify an input example.
    For example:

    ```sh
    python3 simple-inference.py --input=<path_to_input_jpg>
    ```
