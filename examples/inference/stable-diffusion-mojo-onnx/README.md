# Stable Diffusion inference with Mojo

This directory illustrates how to run Stable Diffusion through MAX Engine.
Specifically, this example extracts StableDiffusion-1.5 from Hugging Face and executes
it via the MAX Engine Mojo API.

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

## Custom Images

Getting started with your own creative prompts is as simple as:

```sh
./text_to_image.🔥 --prompt "my image description" -o my-image.png
```

But of course, there are some additional settings that can be tweaked for more
fine-grained control over image output. See `./text_to_image.🔥 --help` for
details.

## Files

- `download-model.py`: Downloads [runwayml/stable-diffusion-v1-5
](https://huggingface.co/runwayml/stable-diffusion-v1-5)
and exports it as ONNX.

- `text_to_image.🔥`: Example program that runs full stable-diffusion pipeline
through MAX Engine in order to generate images from the given prompt.
