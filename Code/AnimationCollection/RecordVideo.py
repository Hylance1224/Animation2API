# https://blog.csdn.net/zzzzjh/article/details/80903597
"""python + opencv 实现屏幕录制_by-_Zjh_"""
from PIL import ImageGrab
import numpy as np
import cv2
import time

def record_video(path):
    width = 712
    height = 1282
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    video = cv2.VideoWriter(path +'.mp4', fourcc, 6, (width, height))  # 输出文件命名为test.mp4,帧率为16，可以自己设置
    start_time = time.time()
    while True:
        im = ImageGrab.grab(bbox=(0, 0, width, height))
        imm = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
        video.write(imm)
        jiange = int(time.time()) - int(start_time)
        # print(jiange)
        if jiange == 5:
            break
    video.release()
    cv2.destroyAllWindows()


def record_screenshots(d, path):
    images = []
    tmp_path = path + 'tmp/'
    print(tmp_path)
    import os
    os.makedirs(tmp_path)
    for i in range(25):
        image = d.screenshot(format='raw')
        images.append(image)
    for i in range(25):
        open(tmp_path + str(i) + ".jpg", "wb").write(images[i])

# if __name__ == '__main__':
#     record_video("G://1")

