import torch
import random
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from object_detection import RGBDetection
from neural_rendering_utils import NeuralRenderingUtils
from pytorch3d.io import load_objs_as_meshes
from torch.utils.data import Dataset
from pytorch3d.structures import join_meshes_as_scene

from tqdm import tqdm

device = torch.device('cuda')


class MeshesDataset(Dataset):
    
    def __init__(self, blue_cube_file: str, green_cube_file: str, red_cone_file: str,
                 torch_device, N: int = 100, img_size: int = 256, camera_dist: float = 18.0, camera_azim: float = 0.0,
                 camera_elev: float = 0.0):
        super(MeshesDataset, self).__init__()
        self.device = torch_device
        self.random_translations = torch.tensor([[random.randint(-6, 6) for _ in range(9)] for _ in range(N)])
        self.utils = NeuralRenderingUtils(cam_dist=camera_dist, cam_azim=camera_azim, cam_elev=camera_elev,
                                          image_size=img_size)
        self.images = []
        self.coordinates = []
        self.R, self.T = self.utils.get_R_T()
        self.renderer_rgb, self.renderer_silhouette = self.utils.get_renderers()

        # generate the data
        for idx, random_translation in tqdm(enumerate(self.random_translations)):

            # load meshes, translate each one randomly and join them to one mesh
            meshes = load_objs_as_meshes([blue_cube_file, green_cube_file, red_cone_file], device=self.device)
            meshes_pred = meshes.clone()
            meshes_pred.verts_list()[0] += random_translation[torch.randperm(9)][:3].to(self.device).clone()
            meshes_pred.verts_list()[1] += random_translation[torch.randperm(9)][:3].to(self.device).clone()
            meshes_pred.verts_list()[2] += random_translation[torch.randperm(9)][:3].to(self.device).clone()
            meshes_joined = join_meshes_as_scene(meshes_pred, include_textures=True)

            # apply gaussian filter on each image
            gaussian_filter = self.utils.get_gaussian_filter()
            rgb_image = self.renderer_rgb(meshes_world=meshes_joined, R=self.R, T=self.T).to(device)
            sil_image = self.renderer_silhouette(meshes_world=meshes_joined, R=self.R, T=self.T).to(device)
            rgb_blurred = gaussian_filter(rgb_image.permute(0, 3, 1, 2))

            # predict RGB real world coordinates
            object_detection = RGBDetection(img_as_ubyte(rgb_image.cpu()[0].numpy()))
            green_coord, blue_coord, red_coord = object_detection.get_coordinates()
            green_translation, blue_translation, red_translation = \
                self.utils.get_predicted_translations(green_coord, red_coord, blue_coord)

            # translate the meshes according to the prediction
            meshes_pred.verts_list()[0] += blue_translation
            meshes_pred.verts_list()[1] += green_translation
            meshes_pred.verts_list()[2] += red_translation

            # for each iteration add the rgb image, silhouette image, blurred image and predicted mesh
            self.images.append((rgb_image[0].permute(2, 0, 1),
                                sil_image[0].permute(2, 0, 1),
                                rgb_blurred[0],
                                meshes_pred))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item]


# sphere_file = '/home/nadav2/Downloads/Cube1.obj'
# cube_file = '/home/nadav2/Downloads/Cube0.obj'
# cone_file = '/home/nadav2/Downloads/Cone.obj'
#
# meshes = MeshesDataset(blue_cube_file=cube_file, red_cone_file=cone_file, green_cube_file=sphere_file,
#                        torch_device=device)


def show_dataset(meshes, imgs_in_row, num_of_rows):
    imgs_collage = [torch.cat([meshes.__getitem__(i)[0].cpu().permute(1, 2, 0) for i in
                               range(i * imgs_in_row, (i + 1) * imgs_in_row)], 1)
                    for i in range(num_of_rows)]
    plt.imshow(torch.cat(imgs_collage))
    plt.xlim(0, 256 * imgs_in_row)
    plt.ylim(0, 256 * num_of_rows)
    for line in range(1, imgs_in_row):
        plt.vlines(256 * line, 0, 256 * num_of_rows)
    for line in range(1, num_of_rows):
        plt.hlines(256 * line, 0, 256 * imgs_in_row)
    plt.show()

# show_dataset(meshes, 10, 10)/