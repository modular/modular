#!/usr/bin/env python3

import argparse, os

from openai import OpenAI


def test_text(model_name: str, base_api_url: str):
    print("-- testing text request")
    client = OpenAI(base_url=base_api_url, api_key="not_required")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "What is trustable software?"
            },
        ],
    )
    print(response.choices[0].message.content)


def test_vision(model_name: str, base_api_url: str, image_url: str = "https://gift-a-tree.com/wp-content/uploads/2021/10/Oak-tree-1.jpg"):
    print(f"-- testing vision request with sample image from {image_url}")
    client = OpenAI(base_url=base_api_url, api_key="not_required")
    response = client.chat.completions.create(
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
    )
    print(response.choices[0].message.content)


# https://docs.modular.com/max/max-cli#max-serve
def serve(model_name: str, weight_path: str, custom_architecture: str, serve_local: bool = False):
    print(f"-- serving {model_name}")
    cmd = f"max serve --model {model_name}"
    if not serve_local: # if running on DevHW, use GPU
        print("ssh in and do something")
        # max warm-cache --model {model_name} # ?? might be useful
        # max serve --model {model_name} --devices=gpu 
    else: # if running locally, use CPU
        cmd += " --devices=cpu"
        if weight_path:
            cmd += f" --weight-path {weight_path}"
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
        default='8001',
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
        serve(model_name=args.model, weight_path=args.weights, custom_architecture=args.architecture, serve_local=True)


if __name__ == "__main__":
    main()
