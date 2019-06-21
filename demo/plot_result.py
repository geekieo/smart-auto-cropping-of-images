'''
裁图结果展示
6幅图像为1组。  
左上：原图           中上：热力图+裁图框    右上：热力图+非极大值约束裁图框
左下：传统算法结果    中下：新算法结果       右下：新算法+非极大值约束结果
'''
import sys
sys.path.append("D:\\PySpace\\SACI")

from utils.io import get_image_list
from utils.io import imread
import matplotlib.pyplot as plt 

src_list = get_image_list('D:\dataset\crop_image\deepgazeII\src')
fusion_win_list = get_image_list('D:\dataset\crop_image\deepgazeII\Result_20180601185154')
fusion_win_r_list = get_image_list('D:\dataset\crop_image\deepgazeII\Result_20180629160645')
legacy_crop_list = get_image_list('D:\dataset\crop_image\deepgazeII\Result_20180601114526')
dgII_crop_list = get_image_list('D:\dataset\crop_image\deepgazeII\Result_20180601143635')
dgII_crop_r_list = get_image_list('D:\dataset\crop_image\deepgazeII\Result_20180629160733')


def plot(img1, img2, img3, img4, img5, img6):
    plt.figure(figsize=(13,7)) # 1300*700
    
    plt.subplot(231)
    plt.title('src')
    plt.axis('off')
    plt.imshow(img1)
    
    plt.subplot(232)
    plt.title('deepgazeII + crop window')
    plt.axis('off')
    plt.imshow(img2)

    plt.subplot(233)
    plt.title('deepgazeII + crop window with restrain')
    plt.axis('off')
    plt.imshow(img3)
    
    plt.subplot(234)
    plt.title('legacy result')
    plt.axis('off')
    plt.imshow(img4)
    
    plt.subplot(235)
    plt.title('deepgazeII result')
    plt.axis('off')
    plt.imshow(img5)
    
    plt.subplot(236)
    plt.title('deepgazeII with restrain result ')
    plt.axis('off')
    plt.imshow(img6)

    plt.show()


for i in range(len(src_list)):
    src = imread(src_list[i])
    fusion_win = imread(fusion_win_list[i])
    fusion_win_r = imread(fusion_win_r_list[i])
    legacy_crop = imread(legacy_crop_list[i])
    dgII_crop = imread(dgII_crop_list[i])
    dgII_crop_r = imread(dgII_crop_r_list[i])
    plot(src, fusion_win, fusion_win_r, legacy_crop,dgII_crop,dgII_crop_r)