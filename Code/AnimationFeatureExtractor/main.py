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

from lib import ParseGRU,Visualizer
from network import ThreeD_conv


# 教程 https://qiita.com/satolab/items/09a90d4006f46e4e959b


def transform(video):
    # video has (depth,frame,img,img)
    trans_video = torch.empty(opt.n_channels,opt.T,opt.image_height,opt.image_width)
    for i in range(opt.T):
        img = video[:,i]
        img = trans(img).reshape(opt.n_channels,opt.image_height,opt.image_width)
        trans_video[:,i] = img
    return trans_video


def trim(video):
    start = np.random.randint(0, video.shape[1] - (opt.T+1))
    end = start + opt.T
    return video[:, start:end, :, :]


def random_choice(n_videos, files):
    X = []
    for _ in range(opt.batch_size):
        # ndrray类型
        file = files[np.random.randint(0, n_videos-1)]
        video = read_gif(file)
        while video is None:
            file = files[np.random.randint(0, n_videos - 1)]
            video = read_gif(file)
            # print(file)
        video = video.transpose(3, 0, 1, 2) / 255.0
        # trim video 减少帧数
        # video = torch.Tensor(trim(video))  # video has (depth,frame,img,img)
        video = torch.Tensor(video)
        video = transform(video)
        # print(video.shape)
        # (1, 16, 128, 128)
        X.append(video)
    X = torch.stack(X)
    # print(X.shape)
    return X


def get_gif(path):
    gifs = []
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(path+'/'+file):
            gifs.extend(get_gif(path+'/'+file))
        else:
            gifs.append(path+'/'+file)
    return gifs


def read_gifs(files):
    videos = []
    for file in files:
        img = Image.open(file)
        a_frames = []
        i = 1
        final_frame = None
        for frame in ImageSequence.Iterator(img):
            # Converting it to RGB to ensure that it has 3 dimensions as requested
            frame = frame.convert('RGB')
            a_frames.append(numpy.asarray(frame))
            final_frame = frame
            if i >= opt.frame:
                break
            i = i + 1
        if i < opt.frame:
            for w in range(opt.frame - i + 1):
                a_frames.append(final_frame)
        try:
            a = numpy.stack(a_frames)
        except:
            pass
            # print(file)
        if a.shape == (16, 500, 281, 3):
            videos.append(a)
    return videos


def read_gif(file):
    img = Image.open(file)
    a_frames = []
    i = 1
    final_frame = None
    for frame in ImageSequence.Iterator(img):
        # Converting it to RGB to ensure that it has 3 dimensions as requested
        frame = frame.convert('RGB')
        a_frames.append(numpy.asarray(frame))
        final_frame = frame
        if i >= opt.frame:
            break
        i = i + 1
    if i < opt.frame:
        for w in range(opt.frame - i + 1):
            a_frames.append(final_frame)
    try:
        a = numpy.stack(a_frames)
    except:
        return None
        # print(file)
    if a.shape == (16, 500, 281, 3):
        return a
    return None


if __name__ == '__main__':
    parse = ParseGRU()
    opt = parse.args
    autoencoder = ThreeD_conv(opt)
    autoencoder.train()  # 设定训练模式
    mse_loss = nn.MSELoss()  # 创建一个标准，测量输入xx和目标yy中每个元素之间的均方误差
    #  优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
    optimizer = torch.optim.Adam(autoencoder.parameters(),
                                 lr=opt.learning_rate,
                                 weight_decay=1e-5)

    # files = glob.glob(opt.dataset + '/*')  # 获取文件夹下所有文件
    # videos = [skvideo.io.vread(file) for file in files]

    files = get_gif("G:\\animations\\ah.creativecodeapps.tiempo")
    # files = get_gif("G:\\animations")
    # videos = read_gifs(files)
    #
    # # ndarray of 4-dimension (number of frame, height, width, depth)
    # videos = [video.transpose(3, 0, 1, 2) / 255.0 for video in videos]
    # # 转置 ndarray of 4-dimension (depth, number of frame, height, width)
    n_videos = len(files)
    print(n_videos)
    if opt.cuda:
        autoencoder.cuda()
    # transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.Resize((opt.image_height, opt.image_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    losses = np.zeros(opt.n_itrs)
    visual = Visualizer(opt)

    for itr in range(opt.n_itrs):
        real_videos = random_choice(n_videos, files)
        x = real_videos
        print(x.shape)
        print('-------------')
        if opt.cuda:
            x = Variable(x).cuda()
        else:
            x = Variable(x)

        xhat = autoencoder(x)
        # print(x.shape)
        # print(xhat.shape)
        # print('----')
        loss = mse_loss(xhat, x)
        losses[itr] = losses[itr] * (itr / (itr + 1.)) + loss.data * (1. / (itr + 1.))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('itr [{}/{}], loss: {:.4f}'.format(
            itr + 1,
            opt.n_itrs,
            loss))
        visual.losses = losses
        visual.plot_loss()
        if itr % 2000 == 0:
            state = {'model': autoencoder.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, 'logs/model/model'+str(itr))
        if itr % opt.check_point == 0:
            tests = x[:opt.n_test].reshape(-1, opt.T, opt.n_channels, opt.image_height, opt.image_width)
            recon = xhat[:opt.n_test].reshape(-1, opt.T, opt.n_channels, opt.image_height, opt.image_width)

            for i in range(opt.n_test):
                # if itr == 0:
                save_image((tests[i] / 2 + 0.5), os.path.join(opt.log_folder + '/generated_videos/3dconv',
                                                              "real_itr{}_no{}.png".format(itr, i)))
                save_image((recon[i] / 2 + 0.5), os.path.join(opt.log_folder + '/generated_videos/3dconv',
                                                              "recon_itr{}_no{}.png".format(itr, i)))
                # torch.save(autoencoder.state_dict(), os.path.join('./weights', 'G_itr{:04d}.pth'.format(itr+1)))
