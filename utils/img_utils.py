import torch
import numpy as np
import imageio
import cv2

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor * 256
    tensor[tensor > 255] = 255
    tensor[tensor < 0] = 0
    tensor = tensor.type(torch.uint8).permute(1, 2, 0).cpu().numpy()

    return tensor

def get_video():
    video_url = 'inputs/marker.mp4'
    r = 128
    # skip_every = 2
    cap = cv2.VideoCapture(video_url)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = r
    frameHeight = r

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('float32'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, img = cap.read()
        if img is None:
            break
        img = cv2.resize(img, (r,r))
        buf[fc] = img / 255.0
        fc += 1

    # buf = buf[::skip_every, :, :, :]
    # Make frame count divisible by 10
    frameCount = min(frameCount, fc)
    new_frame_count = frameCount - (frameCount % 10)
    buf = buf[:new_frame_count]
    # print("BUF", buf.shape)
    cap.release()
    return buf

def get_image():
    # image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
    image_url = 'inputs/jennie.png'
    img = imageio.imread(image_url)[..., :3] / 255.
    c = [img.shape[0] // 2, img.shape[1] // 2]
    r = 256
    # img = img[c[0] - r:c[0] + r, c[1] - r:c[1] + r] # crop image
    img = cv2.resize(img, (r,r)) # resize entire image
    return img
