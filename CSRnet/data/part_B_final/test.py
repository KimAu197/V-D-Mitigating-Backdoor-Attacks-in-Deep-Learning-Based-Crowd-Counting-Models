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

if __name__=="__main__":
    
    cfg = Config()                                                          # configuration
    model = CSRNet().to(cfg.device)                                         # model
    criterion = nn.MSELoss(size_average=False)                              # objective
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)              # optimizer
    # train_dataloader = create_train_dataloader(cfg.dataset_root, use_flip=True, batch_size=cfg.batch_size)
    test_dataloader  = create_test_dataloader(cfg.dataset_root)             # dataloader
    checkpoint_file = "/root/backdoor/CSRNet-pytorch-master/checkpoints/100_add_ft.pth"
    # loaded_model = torch.load('./parm/model_params.pth')


    # loaded_params = torch.load('model_params.pth')

    # model.load_state_dict(loaded_model)


    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)
    min_mae = sys.maxsize
    min_mae_epoch = -1
    
    file2 = "output3"
    csv_file_path = os.path.join("./test_data/", file2, "data_100.csv")


    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)


    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'gt_density_sum', 'et_density_sum', 'rate'])
            
    epoch_mae_min = 100000
    epoch_mae_max = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            # import pdb
            # pdb.set_trace()
            image = data['image'].to(cfg.device)
            gt_densitymap = data['densitymap'].to(cfg.device)
            et_densitymap = model(image).detach()           # forward propagation
            mae = abs(et_densitymap.data.sum()-gt_densitymap.data.sum())
            if mae <= epoch_mae_min:
                epoch_mae_min = mae
            if mae >= epoch_mae_max:
                epoch_mae_max = mae

            # save_image(image, "./test_data/"+ file2 + "/real_image/{}.jpg".format(i))
            
            # save_image(et_densitymap[0]/torch.max(et_densitymap[0]), "./test_data/" + file2 + "/et_densitymap/{}.jpg".format(i))
            # save_image(gt_densitymap[0]/torch.max(et_densitymap[0]), "./test_data/" + file2 + "/gt_densitymap/{}.jpg".format(i))
            gtd = gt_densitymap[0].cpu().sum().tolist()
            etd = et_densitymap[0].cpu().sum().tolist()
            data = [i, gtd, etd, etd/gtd]

            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
                

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch_mae_min', epoch_mae_min])
            writer.writerow(['epoch_mae_max', epoch_mae_max])
# %%
