#%%
import numpy as np
import time
import torch
import torch.nn as nn
import os, csv
import sys
from tqdm import tqdm
import json
from config import Config
from model import CSRNet
from dataset import create_train_dataloader,create_test_dataloader
from utils import denormalize
from torchvision.utils import save_image
import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch import Tensor
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image

if __name__=="__main__":
    
    cfg = Config()                                                          # configuration
    model = CSRNet().to(cfg.device)                                         # model
    criterion = nn.MSELoss(size_average=False)                              # objective
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)              # optimizer
    # train_dataloader = create_train_dataloader(cfg.dataset_root, use_flip=True, batch_size=cfg.batch_size)
    test_dataloader  = create_test_dataloader(cfg.dataset_root)             # dataloader
    checkpoint_file = "./checkpoints/clean.pth"
    # loaded_model = torch.load('./parm/model_params.pth')


    # loaded_params = torch.load('model_params.pth')

    # model.load_state_dict(loaded_model)

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)

    def forward_hook(module, inp, outp):     # Define forward hook
        feature_map.append(outp)             # Append the output to feature_map

    def backward_hook(module, inp, outp):    # Define backward hook
        grad.append(outp)                    # Append the output to grad

    
    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
    ])
    def _normalize(cams: Tensor) -> Tensor:
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

        return cams
    
    def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
        """Overlay a colormapped mask on a background image

        Args:
            img: background image
            mask: mask to be overlayed in grayscale
            colormap: colormap to be applied on the mask
            alpha: transparency of the background image

        Returns:
            overlayed image
        """

        if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
            raise TypeError('img and mask arguments need to be PIL.Image')

        if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
            raise ValueError('alpha argument is expected to be of type float between 0 and 1')

        cmap = cm.get_cmap(colormap)    
        # Resize mask and apply colormap
        overlay = mask.resize(img.size, resample=Image.BICUBIC)
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
        # Overlay the image with the mask
        overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

        return overlayed_img
        
    file2 = "output_clean"
    csv_file_path = os.path.join("./test_data/", file2, "data.csv")
    model.eval()

    # os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    # with open(csv_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['index', 'gt_density_sum', 'et_density_sum', 'poisoned', 'rate'])
        
    # for i, data in enumerate(tqdm(test_dataloader)):
    #     # import pdb
    #     # pdb.set_trace()
    #     image = data['image'].to(cfg.device)
    #     image = image.requires_grad_()
    #     gt_densitymap = data['densitymap'].to(cfg.device)
    #     et_densitymap = model(image).detach()
    #     gtd = gt_densitymap[0].cpu().sum().tolist()
        # etd = et_densitymap[0].cpu().sum().tolist()    
            
    for i in range(1,317):
        feature_map = []  
        grad = []  

        
        handle_forward = model.output_layer.register_forward_hook(forward_hook)
        handle_backward = model.output_layer.register_full_backward_hook(backward_hook)
        
        img_path = "/root/backdoor/CSRNet-pytorch-master/data/part_B_final/test_data/images/IMG_{}.jpg".format(i)
        den_path = "/root/backdoor/CSRNet-pytorch-master/data/part_B_final/test_data/densitymaps/IMG_{}.npy".format(i)
         
        density_map = np.load(den_path)
       
        gtd = density_map.sum()
        orign_img = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB
        img = preprocess(orign_img)                      # Preprocess the image
        img = torch.unsqueeze(img, 0)                    # Add batch dimension [1, 3, 224, 224]
        # Ensure the input tensor requires gradients
        img = img.cuda().requires_grad_()
        # Forward pass
        et_densitymap = model(img)
        etd = et_densitymap.sum()
        model.zero_grad()
        etd.backward(retain_graph=True)
        
        weights = grad[0][0].squeeze(0).mean(dim=(1, 2))      
        grad_cam = (weights.view(*weights.shape, 1, 1) * feature_map[0].squeeze(0)).sum(0)
        grad_cam = _normalize(F.relu(grad_cam, inplace=True)).cpu()
        mask = to_pil_image(grad_cam.detach().numpy(), mode='F')
        
        
        result = overlay_mask(orign_img, mask) 
        result.save("/root/backdoor/CSRNet-pytorch-master/test_data/output_clean/image/{}.jpg".format(i))
        img2 = preprocess(result)                      # Preprocess the image
        img2 = torch.unsqueeze(img2, 0)                    # Add batch dimension [1, 3, 224, 224]
        img2 = img2.cuda().requires_grad_()

        # Forward pass
        et_densitymap2 = model(img2)

        etd2 = et_densitymap2.sum()
        data = [i, gtd, etd.item(), etd2.item(), (etd2/gtd).item()]
        print(data)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
       
     
        handle_forward.remove()
        handle_backward.remove()

        
