<!-- markdownlint-disable -->

### **MAX base container**

[Modular Accelerated eXecution (MAX)⁠](https://docs.modular.com/max/) provides a high-performance, flexible platform for AI workloads, leveraging modern GPUs to deliver accelerated generative AI performance while maintaining portability across different hardware configurations and cloud providers.

The MAX base container (`max-nvidia-base`) provides a lightweight environment optimized for AI model deployment with minimal dependencies. It includes essential components such as CUDA and PyTorch (CPU) while omitting heavier frameworks like cuDNN.

This container is ideal for users who need a streamlined solution with faster downloads and a smaller footprint. For more information on container contents and instance compatibility, see [MAX containers⁠](https://docs.modular.com/max/container/) in the MAX documentation.

### **Quick Start**

You can run an LLM on GPU using the latest MAX base container with the following command:

```
docker run --gpus 1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_HUB_ENABLE_HF_TRANSFER=1" \
    --env "HF_TOKEN=<secret>" \
    -p 8000:8000 \
    modular/max-nvidia-base:<version> \
    --model-path <model-provider/model-id>`
```

You can run a [MAX-optimized model⁠](https://huggingface.co/modularai)⁠ by referencing its Hugging Face model ID. For example, `modularai/Llama-3.1-8B-Instruct-GGUF`.

For more information on quickly deploying popular models with MAX, see [MAX Builds⁠](https://builds.modular.com/).

### **Tags**

Supported tags are updated to the latest MAX versions, which include the latest stable release and more experimental nightly releases. The `latest` tag provides you with the latest stable version and the `nightly` tag provides you with the latest nightly version.

Stable

- max-nvidia-base:25.X

Nightlies

- max-nvidia-base:25.X.0.devYYYYMMDD

### **Documentation**

For more information on Modular and its products, visit the [Modular documentation site⁠](https://docs.modular.com/).

### **Community**

To stay up to date with new releases, [sign up for our newsletter⁠](https://www.modular.com/modverse#signup), [check out the community⁠](https://www.modular.com/community), and [join our forum⁠](https://forum.modular.com/).

If you're interested in becoming a design partner to get early access and give us feedback, please [contact us⁠](https://www.modular.com/company/contact).

### **License**

This container is released under the [NVIDIA Deep Learning Container license⁠](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf?t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9).
