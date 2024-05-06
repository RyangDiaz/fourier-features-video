import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from utils.img_utils import tensor_to_numpy
from utils.visualize_utils import save_numpy_to_video

def create_model(input_dim=2, output_dim=3, num_features=256, device='cpu'):
  model = nn.Sequential(
        nn.Conv2d(
            input_dim,
            num_features,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(num_features),

        nn.Conv2d(
            num_features,
            num_features,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(num_features),

        nn.Conv2d(
            num_features,
            num_features,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(num_features),

        nn.Conv2d(
            num_features,
            output_dim,
            kernel_size=1,
            padding=0),
        nn.Sigmoid(),

    ).to(device)

  return model

class GaussianFourierFeatureTransform(nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, sampling_dist="gaussian", mapping_size=256, scale=10):
        super().__init__()
        assert sampling_dist in ["gaussian", "uniform"]

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size

        if sampling_dist == "gaussian":
          self._B = torch.randn((num_input_channels, mapping_size)) * scale
        elif sampling_dist == "uniform":
          self._B = torch.rand((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

def train_model(
    train_x, 
    train_y, 
    test_x=None, 
    test_y=None, 
    mode="mlp", 
    hidden_layer_size=256, 
    ff_num_features=256, 
    num_steps=400, 
    scale=10, 
    device='cpu',
    save_imgs=True,
    visualize_train=False
):
  assert mode in ["mlp", "ff_gaussian", "ff_uniform"]

  input_dim = train_x.shape[1]
  output_dim = train_y.shape[1]

  if mode == "ff_gaussian":
    train_x = GaussianFourierFeatureTransform(
        num_input_channels = input_dim, 
        sampling_dist = "gaussian",
        mapping_size = ff_num_features // 2, 
        scale = scale
    )(train_x)
    input_dim = ff_num_features
  elif mode == "ff_uniform":
    train_x = GaussianFourierFeatureTransform(
      num_input_channels = input_dim, 
      sampling_dist = "uniform",
      mapping_size = ff_num_features // 2, 
      scale = scale
    )(train_x)
    input_dim = ff_num_features

  train_x = train_x.to(device)
  train_y = train_y.to(device)
  # print(train_x.shape, train_y.shape)

  model = create_model(input_dim=input_dim, output_dim=output_dim, num_features=hidden_layer_size, device=device)

  optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)
  model_loss = torch.nn.MSELoss()
  model_psnr = lambda loss: -10 * np.log10(loss.item())

  train_losses = []
  train_psnrs = []
  pred_imgs = []
  xs = []

  for epoch in tqdm(range(num_steps+1)):
      optimizer.zero_grad()

      generated = model(train_x)

      # loss = torch.nn.functional.l1_loss(target, generated)
      loss = model_loss(train_y, generated.double())

      loss.backward()
      optimizer.step()

      if epoch % 10 == 0 and visualize_train:
        train_losses.append(loss.item())
        train_psnrs.append(model_psnr(loss))
        pred_imgs.append(tensor_to_numpy(generated[0]))
        xs.append(epoch)

      if epoch % 100 == 0:
        print('Epoch %d, loss = %.03f' % (epoch, float(loss)))

        if save_imgs:
          output_name = mode if mode == "mlp" else f"{mode}_{scale}"
          output_img = tensor_to_numpy(generated[0])
          plt.imsave(f'outputs/{output_name}_{epoch}.png', output_img)

  return {
    'train_losses': train_losses,
    'train_psnrs': train_psnrs,
    'pred_imgs': np.stack(pred_imgs),
    'xs': xs,
  }

def create_model_video(input_dim=3, output_dim=3, num_features=256, device='cpu'):
  model = nn.Sequential(
        nn.Conv3d(
            input_dim,
            num_features,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        nn.BatchNorm3d(num_features),

        nn.Conv3d(
            num_features,
            num_features,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        nn.BatchNorm3d(num_features),

        nn.Conv3d(
            num_features,
            num_features,
            kernel_size=1,
            padding=0),
        nn.ReLU(),
        nn.BatchNorm3d(num_features),

        nn.Conv3d(
            num_features,
            output_dim,
            kernel_size=1,
            padding=0),
        nn.Sigmoid(),

    ).to(device)

  return model

class GaussianFourierFeatureTransformVideo(nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, sampling_dist="gaussian", mapping_size=256, scale=10):
        super().__init__()
        assert sampling_dist in ["gaussian", "uniform"]

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size

        if sampling_dist == "gaussian":
          self._B = torch.randn((num_input_channels, mapping_size)) * scale
        elif sampling_dist == "uniform":
          self._B = torch.rand((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 5, 'Expected 5D input (got {}D input)'.format(x.dim())

        batches, channels, width, height, time = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H, T] to [(B*W*H*T), C].
        x = x.permute(0, 2, 3, 4, 1).reshape(batches * width * height * time, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H*T), C] to [B, W, H, T, C]
        x = x.view(batches, width, height, time, self._mapping_size)
        # From [B, W, H, T, C] to [B, C, W, H, T]
        x = x.permute(0, 4, 1, 2, 3)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

def train_model_video(
    train_x, 
    train_y, 
    test_x=None, 
    test_y=None, 
    mode="mlp", 
    hidden_layer_size=256, 
    ff_num_features=256, 
    num_steps=400, 
    scale=10, 
    device='cpu',
    save_imgs=True,
    visualize_train=True
):
  assert mode in ["mlp", "ff_gaussian", "ff_uniform"]

  input_dim = train_x.shape[1]
  output_dim = train_y.shape[1]

  fourier_features = nn.Identity()

  if mode == "ff_gaussian":
    fourier_features = GaussianFourierFeatureTransformVideo(
        num_input_channels = input_dim, 
        sampling_dist = "gaussian",
        mapping_size = ff_num_features // 2, 
        scale = scale
    )
    train_x = fourier_features(train_x)
    input_dim = ff_num_features
  elif mode == "ff_uniform":
    fourier_features = GaussianFourierFeatureTransformVideo(
      num_input_channels = input_dim, 
      sampling_dist = "uniform",
      mapping_size = ff_num_features // 2, 
      scale = scale
    )
    train_x = fourier_features(train_x)
    input_dim = ff_num_features

  train_x = train_x.to(device)
  train_y = train_y.to(device)
  # print(train_x.shape, train_y.shape)

  model = create_model_video(input_dim=input_dim, output_dim=output_dim, num_features=hidden_layer_size, device=device)

  optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)
  model_loss = torch.nn.MSELoss()
  model_psnr = lambda loss: -10 * np.log10(loss.item())

  train_losses = []
  train_psnrs = []
  pred_imgs = []
  xs = []

  for epoch in tqdm(range(num_steps+1)):
      optimizer.zero_grad()

      generated = model(train_x)
      loss = model_loss(train_y.double(), generated.double())

      loss.backward()
      optimizer.step()

      if epoch % 10 == 0 and visualize_train:
        train_losses.append(loss.item())
        train_psnrs.append(model_psnr(loss))
        pred_imgs.append(tensor_to_numpy(generated[0,...,0]))
        xs.append(epoch)

      if epoch % 100 == 0:
        print('Epoch %d, loss = %.03f' % (epoch, float(loss)))

        if save_imgs:
          output_name = mode if mode == "mlp" else f"{mode}_{scale}"
          if epoch == 0:
            save_numpy_to_video(train_y[0], f"{output_name}_gt")
          # save_numpy_to_video(generated[0], output_name)
          save_numpy_to_video(generated[0], f"{output_name}_{epoch}")
          # output_img = tensor_to_numpy(generated[0,...,0])
          # plt.imsave(f'outputs/{output_name}_{epoch}.png', output_img)

  if len(pred_imgs) > 0:
    pred_imgs = np.stack(pred_imgs)
  else:
    pred_imgs = None
  
  return {
    'model': nn.Sequential(fourier_features, model),
    'train_losses': train_losses,
    'train_psnrs': train_psnrs,
    'pred_imgs': pred_imgs,
    'xs': xs,
  }