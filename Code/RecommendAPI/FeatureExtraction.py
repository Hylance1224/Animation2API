import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
from PIL import ImageSequence
import numpy
from lib import ParseGRU
from network import ThreeD_conv
import cv2


# 教程 https://qiita.com/satolab/items/09a90d4006f46e4e959b

parse = ParseGRU()
opt = parse.args

# transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(1),
        transforms.Resize((opt.image_height, opt.image_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


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


def choice(file):
    X = []
    video = read_video(file)
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


# def calculate_similar(img1, img2):
#     H1 = cv2.calcHist([img1], [1], None, [256], [0, 256])
#     H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)
#
#     H2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
#     H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
#
#     similar = cv2.compareHist(H1, H2, 0)
#     return similar

# 输入：PIL.Image.Image格式的两张图片。输出：两张图片的相似程度
# 图片相似度的计算通过比较图片直方图完成
def calculate_similar(image1, image2):
    img1 = cv2.cvtColor(numpy.array(image1), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(numpy.array(image2), cv2.COLOR_RGB2BGR)
    H1 = cv2.calcHist([img1], [1], None, [256], [0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)

    H2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)

    similar = cv2.compareHist(H1, H2, 0)
    return similar


def read_video(animation):
    im = Image.open(animation)
    # if im.size != (281, 500):
    #     im = im.resize((281, 500), Image.ANTIALIAS)
    frames = []
    for i in range(im.n_frames-1):
        current = im.tell()
        # im.save(pngDir+'\\'+str(current)+'.png')
        frame = im.convert('RGB')
        frames.append(frame)
        im.seek(current + 1)  # 获取下一帧图片
    a_frames = []
    similar_record = []
    last_frame = None
    # 如果当前gif帧数>设定帧数（8），则需要过滤重复帧
    if len(frames) > 16:
        # for循环中，过滤掉一些重复帧
        for f in frames:
            if last_frame is None:
                a_frames.append(f)
                last_frame = f
            else:
                # 比较帧之间的相似度，若相似度太高则删除某些帧
                similar = calculate_similar(f, last_frame)
                if similar < 0.98:
                    a_frames.append(f)
                    last_frame = f
                    if len(a_frames) == 16:
                        break
                    similar_record.append(similar)
    else:
        a_frames = frames
    # 若动画帧数不足，则让最后一帧重复，直至补全到8帧
    while len(a_frames) < opt.frame:
         a_frames.append(a_frames[-1])
    a = None
    a = numpy.stack(a_frames)
    return a


def get_feature(source):
    checkpoint = torch.load('logs/model/model8000')
    state_dict = checkpoint['model']
    autoencoder = ThreeD_conv(opt)
    autoencoder.load_state_dict(state_dict)

    autoencoder.train()  # 设定训练模式
    mse_loss = nn.MSELoss()  # 创建一个标准，测量输入xx和目标yy中每个元素之间的均方误差
    #  优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
    optimizer = torch.optim.Adam(autoencoder.parameters(),
                                 lr=opt.learning_rate,
                                 weight_decay=1e-5)
    gif = choice(source)
    if opt.cuda:
        autoencoder.cuda()
    if opt.cuda:
        x = Variable(gif).cuda()
    else:
        x = Variable(gif)
    feature = autoencoder(x, 'test')
    return feature


if __name__ == '__main__':
    f = get_feature('G:/output_animation/animation_related_API_photo/3/104.gif')
    print(f)




