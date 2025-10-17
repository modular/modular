#!/usr/bin/env python3

import argparse, os

from openai import OpenAI


def test_text(model_name: str, base_api_url: str):
    #base_api_url += "/chat/completions"
    print("-- testing text request")
    print(f"-- target host: {base_api_url} | {model_name}")
    client = OpenAI(base_url=base_api_url, api_key="not_required")
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "What is trustable software?"
            },
        ],
    )
    print(completion.choices[0].message.content)


def test_vision(model_name: str, base_api_url: str, image_url: str = "https://gift-a-tree.com/wp-content/uploads/2021/10/Oak-tree-1.jpg"):
    print(f"-- testing vision request with sample image from {image_url}")
    client = OpenAI(base_url=base_api_url, api_key="not_required")
    completions = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                ]
            },
        ],
        # reponse_format={"type": "json_object"}
    )
    print(completions.choices[0].message.content)


# https://docs.modular.com/max/max-cli#max-serve
def serve(model_name: str, weight_path: str, custom_architecture: str, serve_local: bool = False):
    print(f"-- serving {model_name}")
    cmd = f"max serve --model {model_name}"
    if custom_architecture:
        cmd += f" --custom-architectures {custom_architecture}"
    else: # if running locally, use CPU
        cmd += " --devices=cpu"
        if weight_path:
            cmd += f" --weight-path {weight_path}"
    print(f"-- running: {cmd}")
    os.system(cmd)


def generate(model_name: str, custom_architecture: str, prompt: str):
    print(f"-- generating with {model_name}, this may take a moment")
    cmd = f"max generate --model {model_name} --prompt \"{prompt}\"" # --max-new-tokens 256 --temperature 0.7"
    if custom_architecture:
        cmd += f" --custom-architectures {custom_architecture}"
    print(f"-- running: {cmd}")
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="serve models and test text or vision requests against them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--text',
        action='store_true',
        help='Run in text generation mode'
    )
    mode_group.add_argument(
        '--vision',
        action='store_true',
        help='Run in vision/multimodal mode'
    )
    mode_group.add_argument(
        '--serve',
        action='store_true',
        help='start the model server on designated host'
    )
    mode_group.add_argument(
        '--generate',
        action='store_true',
        help='run inference and then shut down the model when done'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-3.2-1B-Instruct',
        help='Model name or path (optional)'
    )
    parser.add_argument(
        '--hostname',
        type=str,
        default='0.0.0.0',
        help='hostname of DevHW'
    )
    parser.add_argument(
        '--port',
        type=str,
        default='8000',
        help='hostname of DevHW'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='',
        help='path to model weights (optional)'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='',
        help='name of custom architecture module'
    )

    args = parser.parse_args()

    base_api_url = f"http://{args.hostname}:{args.port}/v1"

    if args.text:
        test_text(model_name=args.model, base_api_url=base_api_url)
    elif args.vision:
        test_vision(model_name=args.model, base_api_url=base_api_url)
    elif args.serve:
        serve(model_name=args.model, weight_path=args.weights, custom_architecture=args.architecture, serve_local=False)
    elif args.generate:
        generate(model_name=args.model, custom_architecture=args.architecture, prompt="What is trustable software?")


if __name__ == "__main__":
    main()
