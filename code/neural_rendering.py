import cv2
import torch
import imageio
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from create_dataset import MeshesDataset
from object_detection import RGBDetection
from pytorch3d.io import load_objs_as_meshes
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures import join_meshes_as_scene
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
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)

# get the running utilities
utils = NeuralRenderingUtils()
gaussian_filter = utils.get_gaussian_filter()
renderer_rgb, renderer_silhouette = utils.get_renderers()
R, T = utils.get_R_T()

# TODO: if "working on single image" else "work on dataset"
# load meshes
green_cube_file = '/home/nadav2/Downloads/Cube1.obj'
blue_cube_file = '/home/nadav2/Downloads/Cube0.obj'
red_cone_file = '/home/nadav2/Downloads/Cone.obj'
meshes_raw = load_objs_as_meshes([blue_cube_file, green_cube_file, red_cone_file], device=device)
meshes_init = meshes_raw.clone()
meshes_init.verts_list()[0] += torch.tensor([-3.0, -4.0, -4.0]).to(device)
meshes_init.verts_list()[1] += torch.tensor([4.0, -4.0, -2.0]).to(device)
meshes_init.verts_list()[2] += torch.tensor([-3.0, 4.0, 4.0]).to(device)
meshes_init = join_meshes_as_scene(meshes_init, include_textures=True)
image_ref_rgb = renderer_rgb(meshes_world=meshes_init, R=R, T=T).to(device)
image_ref_rgb_blurred = gaussian_filter(image_ref_rgb.permute(0, 3, 1, 2))
image_ref_rgb_blurred = image_ref_rgb_blurred.permute(0, 2, 3, 1)
image_ref_sil = renderer_silhouette(meshes_world=meshes_init, R=R, T=T).to(device)
object_detection = RGBDetection(image=img_as_ubyte(image_ref_rgb.cpu()[0].squeeze().detach().numpy()))
green_coord, blue_coord, red_coord = object_detection.get_coordinates()
green_translation, blue_translation, red_translation = utils.get_predicted_translations(green_coord=green_coord,
                                                                                        blue_coord=blue_coord,
                                                                                        red_coord=red_coord)

meshes_raw.verts_list()[0] += red_translation
meshes_raw.verts_list()[1] += green_translation
meshes_raw.verts_list()[2] += blue_translation


# TODO: solve problems with dataset
# dataset = MeshesDataset(blue_cube_file=blue_cube_file, green_cube_file=green_cube_file, red_cone_file=red_cone_file,
#                         torch_device=device, N=1)
# dl = DataLoader(dataset, batch_size=1, shuffle=False)


# create the training model
class Model(nn.Module):

    def __init__(self, rgb_renderer, sil_renderer, torch_device, rotation, translation, meshes):
        super().__init__()
        self.R = rotation
        self.T = translation
        self.meshes = meshes
        self.renderer_rgb = rgb_renderer
        self.renderer_sil = sil_renderer
        self.device = torch_device
        self.conv1 = nn.Conv2d(4, 8, 7)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(288, 60)
        self.fc2 = nn.Linear(60, 9)

    def forward(self, x):
        bs = x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        meshes = self.meshes.clone()
        meshes.verts_list()[0] += x[:, :3]
        meshes.verts_list()[1] += x[:, 3:6]
        meshes.verts_list()[2] += x[:, 6:9]
        meshes = join_meshes_as_scene(meshes, include_textures=True).to(self.device)
        predicted_image_rgb = self.renderer_rgb(meshes_world=meshes, R=self.R, T=self.T)
        predicted_image_sil = self.renderer_sil(meshes_world=meshes, R=self.R, T=self.T)

        return predicted_image_rgb.permute(0, 3, 1, 2), predicted_image_sil.permute(0, 3, 1, 2), meshes


# initialize the model
model = Model(rgb_renderer=renderer_rgb, sil_renderer=renderer_silhouette,
              torch_device=device, rotation=R, translation=T, meshes=meshes_raw).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# TODO: set option to deal with both dataset or single image
# start training loop
loop = tqdm(range(1000))
for epoch in loop:
    optimizer.zero_grad()
    pred_image_rgb, pred_image_sil, pred_mesh = model(image_ref_rgb.permute(0, 3, 1, 2))
    pred_image_rgb_blurred = gaussian_filter(pred_image_rgb)
    loss = 0.3 * criterion(pred_image_rgb_blurred, image_ref_rgb_blurred.permute(0, 3, 1, 2)) + \
           0.7 * criterion(pred_image_sil, pred_image_sil)
    loss.backward()
    optimizer.step()
    loop.set_description(f"loss -> {loss.item():.6f}")
    TensorBoard.add_scalar('Loss', loss, epoch)

    image_rgb = pred_image_rgb.detach().squeeze().cpu().permute(1, 2, 0)
    image_rgb = img_as_ubyte(image_rgb)
    rgb_out = cv2.hconcat((image_rgb, img_as_ubyte(image_ref_rgb.cpu().detach().squeeze())))
    cv2.putText(rgb_out, f"Loss -> {loss.data:.6f}", (156, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))
    cv2.imshow('', cv2.resize(rgb_out, (512, 256)))
    cv2.waitKey(1)

writer.close()
TensorBoard.close()
