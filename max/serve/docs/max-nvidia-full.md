
<!-- markdownlint-disable -->

### **MAX full container**

[Modular Accelerated eXecution (MAX)⁠](https://docs.modular.com/max/) provides a high-performance, flexible platform for AI workloads, leveraging modern GPUs to deliver accelerated generative AI performance while maintaining portability across different hardware configurations and cloud providers.

The MAX full container (`max-nvidia-full`) includes all necessary dependencies for running large AI models efficiently on GPUs. It provides a complete environment with support for PyTorch (GPU), CUDA, and cuDNN, ensuring maximum performance for deep learning workloads. This container is ideal for users who need a fully optimized, out-of-the-box solution for deploying AI models.

The MAX container is compatible with the OpenAI API specification and optimized for deployment on GPUs. For more information on container contents and instance compatibility, see [MAX containers⁠](https://docs.modular.com/max/container/) in the MAX documentation.

### **Quick Start**

You can run an LLM on GPU using the latest MAX full container with the following command:

```
docker run \
  --gpus 1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_HUB_ENABLE_HF_TRANSFER=1" \
  --env "HF_TOKEN=<secret>" \
  -p 8000:8000 \
  modular/max-nvidia-full:<version> \
  --model-path <model-provider/model-id>
```

You can run a [MAX-optimized model⁠](https://huggingface.co/modularai)⁠ by referencing its Hugging Face model ID. For example, `modularai/Llama-3.1-8B-Instruct-GGUF`

You can also use the MAX container to run a variety of LLMs hosted on Hugging Face, such as `Qwen/Qwen2.5-1.5B-Instruct`.

For more information on quickly deploying popular models with MAX, see [MAX Builds⁠](https://builds.modular.com/).

### **Tags**

Supported tags are updated to the latest MAX versions, which include the latest stable release and more experimental nightly releases. The `latest` tag provides you with the latest stable version and the `nightly` tag provides you with the latest nightly version.

Stable

- max-nvidia-full:25.X

Nightlies

- max-nvidia-full:25.X.0.devYYYYMMDD

### **Documentation**

For more information on Modular and its products, visit the [Modular documentation site⁠](https://docs.modular.com/).

### **Community**

To stay up to date with new releases, [sign up for our newsletter⁠](https://www.modular.com/modverse#signup), [check out the community⁠](https://www.modular.com/community), and [join our forum⁠](https://forum.modular.com/).

If you're interested in becoming a design partner to get early access and give us feedback, please [contact us⁠](https://www.modular.com/company/contact).

### **License**

This container is released under the [NVIDIA Deep Learning Container license⁠](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf?t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9).
