import urllib.request
import torch
import torch.nn as nn
import tqdm
import numpy as np
import cv2

import imageio
import tqdm
import matplotlib.pyplot as plt
import os

from utils.train_utils import *
from utils.img_utils import *
from utils.visualize_utils import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "img"
    # print(device)

    if mode == "img":
        # Get an image that will be the target for our model.
        img = get_image()
        target = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).to(device)

        # Create input pixel coordinates in the unit square. This will be the input to the model.
        coords = np.linspace(0, 1, target.shape[2], endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords, coords), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)

    elif mode == "video":
        skip_every = 18
        video = get_video()
        all_frames = torch.tensor(video).unsqueeze(0).permute(0, 4, 2, 3, 1)
        print("Total Video Shape", all_frames.shape)
        target = all_frames[...,::skip_every].to(device)
        # print(target.shape)
        t_coords = np.linspace(0, 1, target.shape[4])
        h_coords = np.linspace(0, 1, target.shape[2])
        w_coords = np.linspace(0, 1, target.shape[3])

        xy_grid = np.meshgrid(h_coords, w_coords, t_coords)
        xy_grid = np.stack(np.meshgrid(h_coords, w_coords, t_coords), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 4, 1, 2, 3).float().contiguous().to(device)
        # print(xy_grid.shape)

    train_fn = train_model if mode == "img" else train_model_video

    # outputs["gaussian"] = train_model_video(
    #     train_x=xy_grid, 
    #     train_y=target, 
    #     mode="ff_gaussian", 
    #     scale=10, 
    #     hidden_layer_size=256, 
    #     ff_num_features=256, 
    #     num_steps=500, 
    #     device=device, 
    #     save_imgs=False
    # )

    outputs = {}
    num_steps = 1000 if mode == "video" else 400
    outputs["mlp"] = train_fn(xy_grid, target, mode="mlp", hidden_layer_size=256, num_steps=num_steps, device=device, save_imgs=False, visualize_train=True)

    sampling_dist = "gaussian"

    if mode == "video":
        for scale in [0.5,1,2,5]:
            outputs[f"ff_{sampling_dist}_{scale}"] = train_fn(
                    train_x=xy_grid, 
                    train_y=target, 
                    mode=f"ff_{sampling_dist}", 
                    scale=scale, 
                    hidden_layer_size=256, 
                    ff_num_features=256, 
                    num_steps=1000, 
                    device=device, 
                    save_imgs=False,
                    visualize_train=True
                )

    elif mode == "img":
        for scale in [1,10,100]:
            outputs[f"ff_{sampling_dist}_{scale}"] = train_fn(
                train_x=xy_grid, 
                train_y=target, 
                mode=f"ff_{sampling_dist}", 
                scale=scale, 
                hidden_layer_size=256, 
                ff_num_features=256, 
                num_steps=400, 
                device=device, 
                save_imgs=False,
                visualize_train=True
            )

    os.makedirs('outputs/visualization', exist_ok=True)
    visualize_train(outputs, mode=mode)
    save_train_video(outputs, mode=mode)

    # use trained model to interpolate video frames!
    if mode == "video":
        t_coords = np.linspace(0, 1, target.shape[4] * skip_every)
        h_coords = np.linspace(0, 1, target.shape[2])
        w_coords = np.linspace(0, 1, target.shape[3])

        xy_grid = np.meshgrid(h_coords, w_coords, t_coords)
        xy_grid = np.stack(np.meshgrid(h_coords, w_coords, t_coords), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 4, 1, 2, 3).float().contiguous().to(device)

        generated = {}
        generated['gt'] = all_frames

        for k in outputs.keys():
            with torch.no_grad():
                model = outputs[k]["model"]
                model.eval()
                generated[k] = model(xy_grid).to('cpu') # offload memory from gpu
                del model
                torch.cuda.empty_cache()

        # put all the frames together!
        # print(outputs["mlp"].shape)
        generated_all = torch.concatenate([generated[n] for n in generated.keys()], axis=-2)
        save_numpy_to_video(generated_all, f"skip_{skip_every}_interpolated_gt")
        # save_numpy_to_video(generated[0], f"ff_{sampling_dist}_{skip_every}_interpolated_output")
        # save_numpy_to_video(all_frames[0], f"mlp_interpolated_gt")
        # save_numpy_to_video(generated[0], f"mlp_interpolated_output")


