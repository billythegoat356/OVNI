# This copyright notice applies to this file only
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import argparse
import numpy as np
import torch
import torchvision
from pathlib import Path
import pycuda.driver as cuda
import PyNvVideoCodec as nvc
from pycuda.autoinit import context
import os
from os.path import join, dirname, abspath
from itertools import zip_longest

# Add the parent directory to Python path in a way that works from any working directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Torch and Torchvision related imports
import torch
from torchvision.utils import save_image
from torchvision.transforms import Normalize
import torchvision.models.detection as models

# Output Display related imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import time

CLASSES = []


# Label class mapping for COCO dataset
import pickle
script_dir = dirname(abspath(__file__))
coco_classes_path = join(script_dir, "coco_classes.pickle")
with open(coco_classes_path, "rb") as f:
    CLASSES = pickle.load(f)

def show(imgs, fig = None, axs = None, im = None):
    '''Function to show batch images. Takes in images as input.'''
    if not isinstance(imgs, list):
        imgs = [imgs]

    img = F.to_pil_image(imgs[0])
    if fig is None or axs is None or im is None:
        plt.ion()
        fig, axs = plt.subplots()
        im = axs.imshow(np.asarray(img), cmap='viridis')
    else:
        im.set_data(np.asarray(img))
        
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        im.set_data(np.asarray(img))
        fig.canvas.flush_events()
        time.sleep(0.1)
    return fig, axs, im


def get_filtered_output(input, confidence_threshold):
    '''Function to filter out boxes based on confidence scores'''
    output = {}
    valid_indices = [i for i, conf in enumerate(input['scores']) if conf >= confidence_threshold]
    
    if not valid_indices:
        print("No boxes above the confidence threshold.")
        return

    # Filter boxes, labels, and confidences
    input_boxes = input["boxes"]
    input_labels = input["labels"]
    input_scores = input["scores"]

    output["boxes"] = input_boxes[valid_indices]
    output["labels"] = filtered_labels = [input_labels[i] for i in valid_indices]
    output["scores"] = [input_scores[i] for i in valid_indices]
    output["labels_confidence_string"] = [f"{CLASSES[label]} {confidence:.2f}" for label, confidence in zip(filtered_labels, output["scores"])]

    return output

def print_and_write(message, file_path):
    '''Function to print and write message '''
    print(message)
    
    with open(file_path, 'a') as file:
        file.write(message + '\n')

def decode_and_detect(enc_file_path, detection_output_file_path, confidence_threshold, display_output, gpu_id):
    '''Function to decode and detect the objects in the video frames.
       This used ThreadedDecoder from PyNvC for batch decoding. ThreadedDecoder
       runs in a separate thread and is launched on instance creation. Its aim
       is to provide better pipelining.
       It uses FasterRCNN(resnet50) model for object detection
    '''
    # Cuda initialization and threaded decoder creation
    try:
        cuda.init()
        cudaDevice = cuda.Device(gpu_id)
        cudaCtx = cudaDevice.retain_primary_context()
        cudaCtx.push()
        cudaStreamNvDec = cuda.Stream()
        decoder = nvc.ThreadedDecoder(enc_file_path, buffer_size=12, gpu_id=gpu_id, cuda_context=cudaCtx.handle,
                                  cuda_stream=cudaStreamNvDec.handle, use_device_memory = True,
                                  output_color_type=nvc.OutputColorType.RGBP)
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        return

    # Loading FasterRCNN model
    model = models.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(torch.device('cuda'))
    model.eval()
    
    fig = None
    axs = None
    axs_im = None
    # Decoder loop
    while True:
        frames = decoder.get_batch_frames(3)
        src_tensor_list = []
        image_list = []
        for frame in frames:
            # Get the interoperable tensor from PyNVC without any copy
            # This tensor is in planar RGB format. That is R of all pixels followed
            # by G of all pixels followed by B of all all pixels
            src_tensor = torch.from_dlpack(frame)
            # Store image for display if needed
            if display_output:
                image_list.append(src_tensor)
            # Model expects float values in [0, 1]
            src_tensor = src_tensor.float() / 255.0
            src_tensor_list.append(src_tensor)

        # Loop break conditions. No more frames to decode
        if len(frames) == 0:
            break
        
        # Display needs images in bytes and 0-255 range.
        if display_output:
            image_for_disp = torch.stack(image_list)
        else:
            image_for_disp = torch.empty(1,1)

        # Run the inference on batch input
        with torch.no_grad():
            outputs = model(src_tensor_list)
        
        # Print the bounding box, associated class label and the confidence score
        for output, im in zip_longest(outputs, image_for_disp):
            filtered_output = get_filtered_output(output, confidence_threshold)
            if filtered_output is None:
                continue
            for box, label, score in zip(filtered_output['boxes'], filtered_output['labels'], filtered_output['scores']):
                print_and_write(f"Box: {box}, Label: {CLASSES[label]}, Score: {score}", detection_output_file_path)
            if display_output:
                print (display_output)
                boxes = draw_bounding_boxes(im, boxes=filtered_output['boxes'], labels=filtered_output['labels_confidence_string'], width = 4)
                fig, axs, axs_im = show(boxes, fig, axs, axs_im)

    cudaCtx.pop()
    print(f"Detection output written to {detection_output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        " This sample application demonstartes Object Detection using PyNvC's ThreadedDecoder. It creates an instance of \
        ThreadedDecoder class, and then loops over requesting for a batch frames in each iteration of the loop. The \
        returned frames are directly interoperable with pytorch with zero copy. The frames are then used for object detection \
        in a FasterRCNN model. The box coordinates, class label and confidence scores are printed."
    )
    parser.add_argument(
        "-i", "--encoded_file_path", type=Path, required=True, help="Encoded video file (read from)")
    parser.add_argument(
        "-o", "--detection_output_file_path", type=Path, help="Detection output file path. Default: <input_file_name>_detection.txt")
    parser.add_argument(
        "-c", "--confidence_threshold", type=float, default=0.8, help="Confidence score above which the bounding box and label is considered a valid detection")
    parser.add_argument(
        "-d", "--display_output", action='store_true', help="Enable output display")
    parser.add_argument(
        "-g", "--gpu_id", type=int, default=0, help="Check nvidia-smi for available GPUs. Default: 0")
    args = parser.parse_args()
    
    # Set default detection_output_file_path if not provided
    if args.detection_output_file_path is None:
        input_path = Path(args.encoded_file_path)
        args.detection_output_file_path = input_path.with_name(f"{input_path.stem}_detection.txt")
    
    print ("Args value: ", args.display_output)
    decode_and_detect(args.encoded_file_path.as_posix(), args.detection_output_file_path.as_posix(), args.confidence_threshold, args.display_output, args.gpu_id)
