import NET
import cfg
import torch
import torch.nn as nn
import dateset
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


#损失函数
con_loss_fn = nn.BCEWithLogitsLoss()
center_loss_fn = nn.BCEWithLogitsLoss()
wh_loss_fn = nn.MSELoss()
cls_loss_fn = nn.CrossEntropyLoss()

def loss_fn(output, target, alpaha):
    output = output.permute(0,2,3,1)
    output = output.reshape(output.size(0),output.size(1),output.size(2),3,-1)

    target = target.to(cfg.device)

    mask_obj = target[...,4] > 0

    output_obj, target_obj = output[mask_obj], target[mask_obj]
    mask_noobj = target[...,4] == 0.
    output_noobj, target_noobj = output[mask_noobj], target[mask_noobj]


    #正样本
    loss_obj_conf = con_loss_fn(output_obj[:,4], target_obj[:,4])
    loss_obj_center = center_loss_fn(output_obj[:,0:2], target_obj[:,0:2])
    loss_obj_wh = wh_loss_fn(output_obj[:,2:4], target_obj[:,2:4])
    loss_obj_cls = cls_loss_fn(output_obj[:,5:], target_obj[:,5].long())

    loss_obj = loss_obj_conf+loss_obj_center+loss_obj_wh+loss_obj_cls

    #负样本
    los_noobj = con_loss_fn(output_noobj[:,4],target_noobj[:,4])

    return alpaha*loss_obj + (1-alpaha)*los_noobj



if __name__ == '__main__':
    data_sets = dateset.MyDataset()
    train_loader = DataLoader(data_sets, batch_size=4, shuffle=True)
    writer = SummaryWriter()

    net = NET.MainNet(cfg.CLASS_NUM).to(cfg.device)

    if os.path.exists(cfg.MODEL_PATH):
        print("netLoad")
        net.load_state_dict(torch.load(cfg.MODEL_PATH))

    # for name, value in net.named_parameters():
    #     if name[:8] == "trunk_13":
    #         print(value)

    opt = torch.optim.Adam(net.parameters())

    loss =None
    for epoch in range(100):
        for target_13,target_26,target_52,imgData in train_loader:
            ouput_13,ouput_26,ouput_52 = net(imgData)

            loss_13 = loss_fn(ouput_13, target_13,0.9)
            loss_26 = loss_fn(ouput_26, target_26,0.9)
            loss_52 = loss_fn(ouput_52, target_52,0.9)

            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()

        print("Loss:",loss)
        writer.add_scalar("loss",loss, global_step=epoch)
        torch.save(net.state_dict(), cfg.MODEL_PATH)









