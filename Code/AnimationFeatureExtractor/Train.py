import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import skvideo.io
skvideo.setFFmpegPath(r'D:/ffmpeg-4.4-essentials_build/bin')
import glob
from PIL import Image
from PIL import ImageSequence
import numpy
import opts
import dataset

from lib import ParseGRU,Visualizer
from ThreeDCNN import ThreeD_conv


# 教程 https://qiita.com/satolab/items/09a90d4006f46e4e959b

if __name__ == '__main__':
    opt = opts.parse_opts()

    # Configure data loader
    data = dataset.Video(opt)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)
    autoencoder = ThreeD_conv(opt)
    autoencoder.train()  # 设定训练模式
    mse_loss = nn.MSELoss()  # 创建一个标准，测量输入xx和目标yy中每个元素之间的均方误差
    #  优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
    optimizer = torch.optim.Adam(autoencoder.parameters(),
                                 lr=opt.learning_rate,
                                 weight_decay=1e-5)

    for epoch in range(opt.n_epochs):
        for i, (imgs) in enumerate(data_loader):
            batch_size = imgs.shape[0]
            x = Variable(imgs)
            print(x.size())
            xhat = autoencoder(x, 'train')
            loss = mse_loss(xhat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss: {:.4f}'.format(loss))





