#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.web_io import urlimread
from predict import DeepGaze
from crop import visualize_crop


deepgaze = DeepGaze()

url_list=[
    r'http://q0.ifengimg.com/2018_45/40F97DFB429B331CBD0CEFCE2BAA4D5AA0FA6DB3_w1080_h10665.jpg',
    r'http://p0.ifengimg.com/pmop/2018/1114/EE85B48F71465531BABEC68653DD9D4107CF7870_size85_w311_h557.jpeg',
    r'http://p0.ifengimg.com/pmop/2018/1110/66A97991D8FBDD28A1A5D1C8B0FC557C1D1441ED_size56_w457_h686.jpeg',
]

for i, url in enumerate(tqdm(url_list)):
    src = urlimread(url)
    if src is None:
        print('WARNING: "{}" is None'.format(url))
        continue
    fig = plt.figure('figure_'+str(i), figsize=(19.0, 10.0))  # in inches!
    
    # src
    plt.subplot(141)
    plt.imshow(src)
    gauss, src_shape = deepgaze.predict(src)
    # saliency map
    plt.subplot(142)
    plt.imshow(gauss)
    # crop
    y, x, h, w = deepgaze.crop(gauss, src_shape,(392, 698),True)   # (312, 698)
    plt.subplot(143)
    plt.imshow(src[y:y+h, x:x+w])
    # 留白 crop 测试 
    ny, nx, nh, nw = deepgaze.crop(gauss, src_shape,(312, 698),True)
    ny -= int((h-nh)/2)
    nh = h
    plt.subplot(144)
    plt.imshow(src[ny:ny+nh, nx:nx+nw])
    # centerbias crop 测试
    

    
    plt.show()