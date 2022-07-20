import cv2
import torch
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage import img_as_ubyte
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene
from neural_rendering_utils import NeuralRenderingUtils


def create_dataset(dist: float, azim: float, elev: float, img_size: int, meshes_dir: str, torch_device: str, N: int):
    torch_device = torch.device(torch_device)
    translations_list = [[4, -4, 0],
                         [-4, -4, 0],
                         [4, 4, 0],
                         [-4, 0, 0],
                         [0, 4, 0],
                         [4, 0, 0],
                         [0, -4, 0],
                         [-4, 4, 0],
                         [0, 0, 0]]
    utils = NeuralRenderingUtils(cam_dist=dist, cam_azim=azim, cam_elev=elev, image_size=img_size)
    R, T = utils.get_R_T()
    renderer_rgb, renderer_silhouette = utils.get_renderers()
    # generate the data
    for i in tqdm(range(N)):

        translations = torch.tensor(random.sample(translations_list, k=7))
        # load meshes, translate each one randomly and join them to one mesh
        meshes_files = [str(path) for path in Path(meshes_dir).rglob('*.obj')]
        meshes_raw = load_objs_as_meshes(meshes_files, device=torch_device)
        meshes_pred = meshes_raw.clone()
        # random_translations = torch.tensor(random.choice(self.translations))
        random_translations = translations[torch.randperm(7)]
        for idx, translation in enumerate(random_translations):
            meshes_pred.verts_list()[idx] += translation.to(torch_device)
        meshes_joined = join_meshes_as_scene(meshes_pred, include_textures=True)

        # apply gaussian filter on each image
        gaussian_filter = utils.get_gaussian_filter()
        rgb_image = renderer_rgb(meshes_world=meshes_joined, R=R, T=T).to(torch_device)
        sil_image = renderer_silhouette(meshes_world=meshes_joined, R=R, T=T).to(torch_device)
        rgb_blurred = gaussian_filter(rgb_image.permute(0, 3, 1, 2))

        iteration_dir = Path(meshes_dir).parent / 'dataset' / str(i)
        if Path.exists(iteration_dir):
            shutil.rmtree(iteration_dir)
        Path.mkdir(iteration_dir)

        rgb_image = img_as_ubyte(np.clip(rgb_image.cpu().squeeze().numpy(), 0, 1))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'/home/nadav2/dev/data/mc-data/Omer/NeuralRendering/dataset/{i}/rgb.png', rgb_image)

        # sil_image = img_as_ubyte(np.clip(sil_image.cpu().squeeze().numpy(), 0, 1))
        sil_image = img_as_ubyte(sil_image[..., 3].cpu().detach().squeeze())
        cv2.imwrite(f'/home/nadav2/dev/data/mc-data/Omer/NeuralRendering/dataset/{i}/sil.png', sil_image)

        rgb_blurred = rgb_blurred.permute(0, 2, 3, 1).cpu()[0]
        rgb_blurred = img_as_ubyte(np.clip(rgb_blurred.cpu().squeeze().numpy(), 0, 1))
        rgb_blurred = cv2.cvtColor(rgb_blurred, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'/home/nadav2/dev/data/mc-data/Omer/NeuralRendering/dataset/{i}/blur.png', rgb_blurred)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dist', type=float, default=18.0, help='camera distance')

    parser.add_argument('-a', '--azim', type=float, default=0.0, help='camera azimut')

    parser.add_argument('-e', '--elev', type=float, default=0.0, help='camera elevation')

    parser.add_argument('-i', '--img_size', type=int, default=256, help='image size')

    parser.add_argument('-m', '--meshes_dir', type=str, help='path to meshes directory',
                        default='/home/nadav2/dev/data/mc-data/Omer/NeuralRendering/meshes')

    parser.add_argument('-t', '--torch_device', type=str, help='torch device (cpu/cuda)',
                        default='cuda')

    parser.add_argument('-n', '--N', type=int, help='desired number of images', default=300)

    return parser.parse_args()


def main(arguments):
    create_dataset(dist=arguments.dist, azim=arguments.azim, elev=arguments.elev, img_size=arguments.img_size,
                   meshes_dir=arguments.meshes_dir, torch_device=arguments.torch_device, N=arguments.N)


if __name__ == '__main__':
    args = parse_args()
    main(args)
