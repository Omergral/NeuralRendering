import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from pytorch3d.renderer import (look_at_view_transform, MeshRenderer,
                                MeshRasterizer, FoVPerspectiveCameras,
                                RasterizationSettings, SoftPhongShader,
                                BlendParams, SoftSilhouetteShader)


class NeuralRenderingUtils:

    def __init__(self, device: str = 'cuda', gaussian_filter_kernelsize: int = 15, gaussian_filter_sigma: int = 3,
                 image_size: int = 256, cam_dist: float = 18.0, cam_azim: float = 0.0, cam_elev: float = 0.0):
        self.device = torch.device(device)
        self.kernel_size = gaussian_filter_kernelsize
        self.gaussian_sigma = gaussian_filter_sigma
        self.image_size = image_size
        self.dist = cam_dist
        self.azim = cam_azim
        self.elev = cam_elev

    def get_gaussian_filter(self):
        rng_seed = 102
        cuda_seed = 102
        np.random.seed(rng_seed)
        random.seed(rng_seed)
        os.environ['PYTHONHASHSEED'] = str(rng_seed)

        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed(cuda_seed)
        torch.cuda.manual_seed_all(cuda_seed)

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (self.kernel_size - 1) / 2.
        variance = self.gaussian_sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(4, 1, 1, 1).to(self.device)

        gaussian_filter = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=self.kernel_size, groups=4, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    def get_renderers(self):
        cameras = FoVPerspectiveCameras(device=self.device)
        blend_params_RGB = BlendParams(sigma=1e-5, gamma=1e-5)
        blend_params_sil = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings_RGB = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.001,
            faces_per_pixel=100)
        raster_settings_sil = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0,
            faces_per_pixel=1)
        renderer_rgb = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings_RGB
            ),
            shader=SoftPhongShader(device=self.device, cameras=cameras, blend_params=blend_params_RGB)
        )
        renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings_sil
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params_sil)
        )

        return renderer_rgb, renderer_silhouette

    def get_camera_position(self):
        return torch.tensor(np.array([self.dist, self.azim, self.elev], dtype=np.float32)).to(self.device)

    def get_R_T(self):
        camera_position = self.get_camera_position()
        return look_at_view_transform(dist=camera_position[0], elev=camera_position[1],azim=camera_position[2],
                                      device=self.device)

    def predict_world_coordinates(self, U, V):
        _, T = self.get_R_T()
        u_v_cor = torch.tensor([[U, V, self.dist - 3, 1]], device=self.device)
        cam_intrinsic_mat = torch.tensor([[self.dist / self.image_size, 0., T[0][0]],
                                          [0., self.dist / self.image_size, T[0][1]],
                                          [0., 0., 1.]], device=self.device)
        cam_intrinsic_mat_inv = cam_intrinsic_mat.inverse()
        cam_intrinsic_mat_inv = torch.cat([cam_intrinsic_mat_inv, torch.tensor([[0., 0., 0.]], device=self.device).T], 1)
        cam_extrinsic_mat = torch.tensor([[-1., 0., 0., T.T[0]],
                                          [0., 1., 0., T.T[1]],
                                          [0., 0., -1., T.T[2]],
                                          [0., 0., 1., 0.]
                                          ], device=self.device)
        cam_extrinsic_mat_inv = cam_extrinsic_mat.inverse()
        world_cor = u_v_cor @ cam_extrinsic_mat_inv @ cam_intrinsic_mat_inv.T
        return world_cor

    def get_initial_primitives_positions(self):
        green_before = self.predict_world_coordinates(120 / self.image_size, 120 / self.image_size)
        red_before = self.predict_world_coordinates(120 / self.image_size, 120 / self.image_size)
        blue_before = self.predict_world_coordinates(120 / self.image_size, 120 / self.image_size)
        return green_before, blue_before, red_before

    def get_predicted_translations(self, green_coord, red_coord, blue_coord):
        green_init, blue_init, red_init = self.get_initial_primitives_positions()
        green_after = self.predict_world_coordinates(green_coord[0] / self.image_size, green_coord[1] / self.image_size)
        red_after = self.predict_world_coordinates(red_coord[0] / self.image_size, red_coord[1] / self.image_size)
        blue_after = self.predict_world_coordinates(blue_coord[0] / self.image_size, blue_coord[1] / self.image_size)
        green_translation = green_init - green_after
        blue_translation = blue_init - blue_after
        red_translation = red_init - red_after
        return green_translation, blue_translation, red_translation
