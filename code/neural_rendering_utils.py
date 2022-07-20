import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from skimage import img_as_ubyte
from object_detection import RGBDetection
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (look_at_view_transform, MeshRenderer,
                                MeshRasterizer, FoVPerspectiveCameras,
                                RasterizationSettings, SoftPhongShader,
                                BlendParams, SoftSilhouetteShader, Materials, PointLights)


class NeuralRenderingUtils:

    def __init__(self, meshes_dir: str = None, device: str = 'cuda', gaussian_filter_kernelsize: int = 15,
                 gaussian_filter_sigma: int = 3, image_size: int = 256, cam_dist: float = 18.0, cam_azim: float = 0.0,
                 cam_elev: float = 0.0):
        self.meshes_dir = '/home/nadav2/dev/data/mc-data/Omer/NeuralRendering/meshes'
        self.device = torch.device(device)
        self.kernel_size = gaussian_filter_kernelsize
        self.gaussian_sigma = gaussian_filter_sigma
        self.image_size = image_size
        self.dist = cam_dist
        self.azim = cam_azim
        self.elev = cam_elev

    def get_gaussian_filter(self):
        # rng_seed = 102
        # cuda_seed = 102
        # np.random.seed(rng_seed)
        # random.seed(rng_seed)
        # os.environ['PYTHONHASHSEED'] = str(rng_seed)
        #
        # torch.manual_seed(rng_seed)
        # torch.cuda.manual_seed(cuda_seed)
        # torch.cuda.manual_seed_all(cuda_seed)

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
        materials = Materials(device=self.device, ambient_color=[[1.0, 1.0, 1.0]], diffuse_color=[[0.0, 0.0, 0.0]],
                              specular_color=[[0.0, 0.0, 0.0]], shininess=0.0)
        lights = PointLights(device=self.device, ambient_color=((1.0, 1.0, 1.0),),
                             specular_color=((0.0, 0.0, 0.0),), location=[[0.0, 0.0, 500.0]])
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
            shader=SoftPhongShader(device=self.device, cameras=cameras, blend_params=blend_params_RGB,
                                   lights=lights, materials=materials)
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
        cam_intrinsic_mat_inv = torch.cat([cam_intrinsic_mat_inv, torch.tensor([[0., 0., 0.]], device=self.device).T],
                                          1)
        cam_extrinsic_mat = torch.tensor([[-1., 0., 0., T.T[0]],
                                          [0., 1., 0., T.T[1]],
                                          [0., 0., -1., T.T[2]],
                                          [0., 0., 1., 0.]
                                          ], device=self.device)
        cam_extrinsic_mat_inv = cam_extrinsic_mat.inverse()
        world_cor = u_v_cor @ cam_extrinsic_mat_inv @ cam_intrinsic_mat_inv.T
        return world_cor

    def get_initial_primitives_positions(self):
        assert self.meshes_dir is not None, 'must set the path to the meshes directory'

        initial_positions = {}
        rgb_rend, _ = self.get_renderers()
        R, T = self.get_R_T()
        for mesh_file in Path(self.meshes_dir).rglob('*.obj'):
            color = mesh_file.parts[-1].replace('.obj', '')
            mesh = load_objs_as_meshes([str(mesh_file)], device=self.device)
            rend_img = rgb_rend(meshes_world=mesh, R=R, T=T).to(self.device)
            object_detection = RGBDetection(img_as_ubyte(np.clip(rend_img.cpu()[0].numpy(), 0, 1)))
            color_coords_dict = object_detection.get_coordinates()
            x_initial, y_initial = color_coords_dict[color][:2]
            color_initial_position = self.predict_world_coordinates(x_initial / self.image_size,
                                                                    y_initial / self.image_size)
            initial_positions[color] = color_initial_position

        return initial_positions

    def get_predicted_translations(self, coords_dict):
        predicted_coords_dict = {}
        initial_positions = self.get_initial_primitives_positions()
        for color, position in coords_dict.items():
            after_translation = self.predict_world_coordinates(position[0] / self.image_size,
                                                               position[1] / self.image_size)
            predicted_coords_dict[color] = initial_positions[color] - after_translation

        return predicted_coords_dict
