from utils.img_utils import get_image, get_video, tensor_to_numpy
import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2

def visualize_train(outputs, mode="img"):
    assert mode in ["img", "video"]

    if mode == "img":
        img = get_image()
    else:
        vid = get_video()
        img = vid[0]
        img = img[...,[2,1,0]]
    # plt.figure(figsize=(8, 4))
    # N = len(outputs)
    # for i, k in enumerate(outputs):
    #     plt.subplot(1, N+1, i+1)
    #     plt.imshow(outputs[k]['pred_imgs'][-1])
    #     plt.title(k)
    # plt.subplot(1, N+1, N+1)
    # plt.imshow(img)
    # plt.title('GT')
    images = np.hstack([img] + [output['pred_imgs'][-1] / 255 for output in outputs.values()])
    plt.imsave('outputs/visualization/image_comparison.png', images)

    # Plot train/test error curves
    plt.figure()
    # plt.subplot(121)
    for i, k in enumerate(outputs):
        plt.plot(outputs[k]['xs'], outputs[k]['train_psnrs'], label=k)
    plt.title('Training PSNRs')
    plt.ylabel('PSNR')
    plt.xlabel('Training iter')
    plt.legend()

    plt.savefig("outputs/visualization/train_psnrs.png")
    print("Saved training PSNRs to outputs/visualization/train_psnrs.png")

    # Plot train/test error curves
    plt.figure()
    # plt.subplot(121)
    for i, k in enumerate(outputs):
        plt.plot(outputs[k]['xs'], outputs[k]['train_losses'], label=k)
    plt.title('Training Losses')
    plt.ylabel('MSE Loss')
    plt.xlabel('Training iter')
    plt.legend()

    plt.savefig("outputs/visualization/train_losses.png")
    print("Saved training losses to outputs/visualization/train_losses.png")

def save_train_video(outputs, mode="img"):
    assert mode in ["img", "video"]

    # Save out video
    if mode == "img":
        all_preds = np.concatenate([outputs[n]['pred_imgs'] for n in outputs], axis=-2)
    else:
        all_preds = np.concatenate([outputs[n]['pred_imgs'][...,[2,1,0]] for n in outputs], axis=-2)
    data8 = all_preds.astype(np.uint8)
    print("DATA8", data8.shape)
    f = 'outputs/visualization/training_convergence.mp4'
    imageio.mimwrite(f, data8, fps=20)
    print("Saved training video to outputs/visualization/training_convergence.mp4")

def save_numpy_to_video(data_array, filename):
    print("Saving video...")

    data_array = np.array([tensor_to_numpy(data_array[...,i].squeeze(0))[...,[2,1,0]] for i in range(data_array.shape[-1])])
    print("DATA", data_array.shape)
    data8 = data_array.astype(np.uint8)
    f = f'outputs/{filename}.mp4'
    imageio.mimwrite(f, data8, fps=20)
