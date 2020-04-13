#!/usr/bin/env python3

import json
import sys
import torch
import urllib

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from torchvision import transforms


def get_online_file():
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    return Path(filename)


def main():
    model = torch.hub.load('pytorch/vision:v0.5.0', 'alexnet', pretrained=True)
    model.eval()

    filename = (
        get_online_file() if args.input_file is None else args.input_file
    )

    # sample execution (requires torchvision)
    input_image = Image.open(str(filename))
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to(args.device)
        model.to(args.device)

    with torch.no_grad():
        output = model(input_batch)

    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])

    # The output has unnormalized scores. To get probabilities, you can run a
    # softmax on it.
    prob = torch.nn.functional.softmax(output[0], dim=0)
    idx = prob.argmax(0)
    # print(prob[idx])

    labels = []
    with open(args.labels_file) as f:
        d = json.load(f)
        labels = [v[1] for k, v in d.items()]

    print(labels[idx])


if __name__ == '__main__':
    parser = ArgumentParser(sys.argv)
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    parser.add_argument('--input-file', type=Path)
    parser.add_argument(
        '--labels-file', type=Path, default='imagenet_class_index.json'
    )

    args = parser.parse_args()
    main()
