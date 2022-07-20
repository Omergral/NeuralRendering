import os
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
from object_detection import RGBDetection
from pytorch3d.io import load_objs_as_meshes
from neural_rendering_utils import NeuralRenderingUtils


class MeshesDataset(Dataset):
    
    def __init__(self, images_dir: str, meshes_dir: str):
        super(MeshesDataset, self).__init__()
        self.images_dir = images_dir
        self.meshes_dir = meshes_dir
        self.meshes_files = [str(path) for path in Path(meshes_dir).rglob('*.obj')]
        self.meshes_order = [path.split('/')[-1].replace('.obj', '') for path in self.meshes_files]
        self.meshes_raw = load_objs_as_meshes(self.meshes_files)
        self.utils = NeuralRenderingUtils(device='cpu')

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, item):
        item_dir = sorted(list(Path(self.images_dir).iterdir()), key=lambda x: int(str(x).split('/')[-1]))[item]
        rgb_img = cv2.cvtColor(cv2.imread(str(item_dir / 'rgb.png')), cv2.COLOR_BGR2RGB)
        sil_img = torch.tensor(cv2.imread(str(item_dir / 'sil.png')))
        blur_img = cv2.cvtColor(cv2.imread(str(item_dir / 'blur.png')), cv2.COLOR_BGR2RGB)
        blur_img_rgb = torch.tensor(blur_img) / 255.0
        translated_mesh = self.get_mesh_translation(rgb_img, self.meshes_raw.clone())
        rgb_img_tensor = torch.tensor(rgb_img) / 255.0
        return rgb_img_tensor, sil_img, blur_img_rgb, translated_mesh.verts_packed(), \
               translated_mesh.faces_packed(), translated_mesh.textures.verts_uvs_padded()

    def get_mesh_translation(self, img, mesh):
        obj_detection = RGBDetection(
            image=img)
        col_coords_dict = obj_detection.get_coordinates()
        pred_coords_dict = self.utils.get_predicted_translations(col_coords_dict)
        for idx, color in enumerate(self.meshes_order):
            mesh.verts_list()[idx] += pred_coords_dict[color]
        return mesh
