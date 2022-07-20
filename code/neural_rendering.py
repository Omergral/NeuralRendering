import cv2
import torch
import imageio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from NR_dataset import MeshesDataset
from object_detection import RGBDetection
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from pytorch3d.structures import join_meshes_as_scene, Meshes
from neural_rendering_utils import NeuralRenderingUtils

# set the device on GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"\nRunning on {device.type}")

# set outputs configurations
TensorBoard = SummaryWriter()
filename_output = "./optimization_demo_2.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.1)

# get the running utilities
utils = NeuralRenderingUtils()
gaussian_filter = utils.get_gaussian_filter()
renderer_rgb, renderer_silhouette = utils.get_renderers()
R, T = utils.get_R_T()

work_on_dataset = True
meshes_directory = '/home/nadav2/dev/data/mc-data/Omer/NeuralRendering/meshes'
images_dir = '/home/nadav2/dev/data/mc-data/Omer/NeuralRendering/dataset'
w = 0.7

# load meshes
meshes_files = [str(path) for path in Path(meshes_directory).rglob('*.obj')]
meshes_order = [path.split('/')[-1].replace('.obj', '') for path in meshes_files]
# meshes_raw = load_objs_as_meshes(meshes_files, device=device)
meshes_raw = None

if work_on_dataset:
    # TODO: solve problems with dataset
    dataset = MeshesDataset(images_dir=images_dir, meshes_dir=meshes_directory)
    dl = DataLoader(dataset, batch_size=2, shuffle=False)

else:
    meshes_init = meshes_raw.clone()
    meshes_init.verts_list()[0] += torch.tensor([4.0, 0.0, 0.0], device=device)
    meshes_init.verts_list()[1] += torch.tensor([-4.0, 0.0, 0.0], device=device)
    meshes_init.verts_list()[2] += torch.tensor([-4.0, 4.0, 0.0], device=device)
    meshes_init.verts_list()[3] += torch.tensor([4.0, 4.0, 0.0], device=device)
    meshes_init.verts_list()[4] += torch.tensor([-4.0, -4.0, 0.0], device=device)
    meshes_init.verts_list()[5] += torch.tensor([4.0, -4.0, 0.0], device=device)
    meshes_init = join_meshes_as_scene(meshes_init, include_textures=True)
    image_ref_rgb = renderer_rgb(meshes_world=meshes_init, R=R, T=T)
    image_ref_rgb_blurred = gaussian_filter(image_ref_rgb.permute(0, 3, 1, 2))
    image_ref_rgb_blurred = image_ref_rgb_blurred.permute(0, 2, 3, 1).to(device)
    image_ref_sil = renderer_silhouette(meshes_world=meshes_init, R=R, T=T).to(device)
    # object_detection = RGBDetection(image=img_as_ubyte(np.clip(image_ref_rgb.cpu()[0].squeeze().detach().numpy(), 0, 1)))
    # color_coords_dict = object_detection.get_coordinates()
    # predicted_coords_dict = utils.get_predicted_translations(color_coords_dict)
    #
    # for idx, color in enumerate(meshes_order):
    #     meshes_raw.verts_list()[idx] += predicted_coords_dict[color]


# create the training model
class Model(nn.Module):

    def __init__(self, rgb_renderer, sil_renderer, torch_device, rotation, translation, num_obj):
        super().__init__()
        self.R = rotation
        self.T = translation
        self.renderer_rgb = rgb_renderer
        self.renderer_sil = sil_renderer
        self.device = torch_device
        self.conv1 = nn.Conv2d(3, 8, 7)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(288, 100)
        self.fc2 = nn.Linear(100, 21)
        self.num_obj = num_obj

    def forward(self, x, mesh, textures):
        verts, faces = mesh
        bs = x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        counter = 3  # divides x to tensors of shape (3,)
        indexer = 116  # divides the vertices to per cube vertices
        for i in range(self.num_obj):
            verts[:, (indexer - 116): indexer, :] += x[:, None, (counter - 3): counter]
            counter += 3
            indexer += 116
        for mesh_idx in range(bs):
            textures_packed = pack_padded_sequence(textures[mesh_idx], lengths=[116, 116, 116, 116, 116, 116, 116],
                                                   batch_first=True).data
            mesh = Meshes(verts=verts[mesh_idx].unsqueeze(0), faces=faces[mesh_idx].unsqueeze(0),
                          textures=TexturesVertex(verts_features=textures_packed.unsqueeze(0)))  # TODO: add for loop to join meshes
            meshes_joined = join_meshes_as_scene(mesh, include_textures=True).to(self.device)
            predicted_image_rgb = self.renderer_rgb(meshes_world=meshes_joined, R=self.R, T=self.T)
            predicted_image_sil = self.renderer_sil(meshes_world=meshes_joined, R=self.R, T=self.T)
        # TODO: modify the images to tensors
        return predicted_image_rgb.permute(0, 3, 1, 2), predicted_image_sil.permute(0, 3, 1, 2), mesh


# initialize the model
model = Model(rgb_renderer=renderer_rgb, sil_renderer=renderer_silhouette,
              torch_device=device, rotation=R, translation=T, num_obj=len(meshes_files)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# TODO: set option to deal with both dataset or single image
if work_on_dataset:
    loop = tqdm(range(1000))

    for epoch in loop:

        loss = 0

        for i, (rgb_img, sil_img, blur_img, verts, faces, textures) in enumerate(dl):

            optimizer.zero_grad()
            mesh = verts.to(device), faces.to(device)
            textures = textures.to(device)
            pred_image_rgb, pred_image_sil, pred_mesh = model(rgb_img.permute(0, 3, 1, 2).to(device), mesh, textures)
            pred_image_rgb_blurred = gaussian_filter(pred_image_rgb)
            loss = w * criterion(pred_image_rgb_blurred, blur_img.permute(0, 3, 1, 2)) + \
                   (1 - w) * criterion(pred_image_sil, sil_img)
            loss.backward()
            optimizer.step()
            loop.set_description(f"loss -> {loss.item():.6f}")
            TensorBoard.add_scalar('Loss', loss.item(), epoch)
            torch.cuda.empty_cache()
            image_rgb = pred_image_rgb.detach().squeeze().cpu().permute(1, 2, 0)
            image_rgb = img_as_ubyte(np.clip(image_rgb, 0, 1))
            rgb_out = cv2.hconcat((image_rgb, rgb_img))
            cv2.putText(rgb_out, f"Loss -> {loss.item():.6f}", (156, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
            writer.append_data(rgb_out)
            rgb_out = cv2.cvtColor(rgb_out, cv2.COLOR_BGR2RGB)
            cv2.imshow('', cv2.resize(rgb_out, (512, 256)))
            cv2.waitKey(1)

else:
    # start training loop
    loop = tqdm(range(100))
    for epoch in loop:
        optimizer.zero_grad()
        pred_image_rgb, pred_image_sil, pred_mesh = model(image_ref_rgb.permute(0, 3, 1, 2))
        pred_image_rgb_blurred = gaussian_filter(pred_image_rgb)
        loss = w * criterion(pred_image_rgb_blurred, image_ref_rgb_blurred.permute(0, 3, 1, 2)) + \
               (1 - w) * criterion(pred_image_sil, pred_image_sil)
        loss.backward()
        optimizer.step()
        loop.set_description(f"loss -> {loss.item():.6f}")
        TensorBoard.add_scalar('Loss', loss.item(), epoch)
        torch.cuda.empty_cache()
        image_rgb = pred_image_rgb.detach().squeeze().cpu().permute(1, 2, 0)
        image_rgb = img_as_ubyte(np.clip(image_rgb, 0, 1))
        img_ref = img_as_ubyte(np.clip(image_ref_rgb.cpu().detach().squeeze(), 0, 1))
        rgb_out = cv2.hconcat((image_rgb, img_ref))
        cv2.putText(rgb_out, f"Loss -> {loss.item():.6f}", (156, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
        writer.append_data(rgb_out)
        rgb_out = cv2.cvtColor(rgb_out, cv2.COLOR_BGR2RGB)
        cv2.imshow('', cv2.resize(rgb_out, (512, 256)))
        cv2.waitKey(1)


writer.close()
TensorBoard.close()
