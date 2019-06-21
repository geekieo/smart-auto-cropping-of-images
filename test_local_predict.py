from predict import DeepGaze
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.io import imread,get_image_list
from crop import visualize_crop


deepgaze = DeepGaze()

src_list = get_image_list('D:\dataset\crop_image\crop_images\ori_images')

def show_test():
    for i, src in enumerate(tqdm(src_list)):
        src = imread(src)
        if src is None:
            print('WARNING: src is None')
            continue
        fig = plt.figure('figure_'+str(i), figsize=(19.0, 10.0))  # in inches!
        # src
        plt.subplot(141)
        plt.imshow(src)
        # saliency map
        gauss, src_shape = deepgaze.predict(src)
        plt.subplot(142)
        # crop
        plt.imshow(gauss)
        y, x, h, w = deepgaze.crop(gauss, src_shape,(392, 698),True)   # (312, 698)
        plt.subplot(143)
        plt.imshow(src[y:y+h, x:x+w])
        y, x, h, w = deepgaze.crop(gauss, src_shape,(392, 698),True)
        # 天地留白 crop
        ny, nx, nh, nw = deepgaze.crop(gauss, src_shape,(312, 698),True)
        ny -= int((h-nh)/2)
        nh = h
        plt.subplot(144)
        plt.imshow(src[ny:ny+nh, nx:nx+nw])
        # centerbias crop 
        
        # plt.savefig('D:/experiment/crop_experiment/' + str(i)+'_area_norm_point4')
        plt.show()
    print('Done.')
# show_test()

def performence_test_with_imread():
    src = imread(src_list[0])
    gauss, src_shape = deepgaze.predict(src)
    print('First blood %s'%str(src_shape))
    print('Initialize complated')

    for i, src in enumerate(tqdm(src_list)):
        src = imread(src)
        if src is None:
            print('WARNING: src is None')
            continue
        gauss, src_shape = deepgaze.predict(src)
        y, x, h, w = deepgaze.crop(gauss, src_shape,(392, 698),True)   # (312, 698) 
        print(y,x,h,w)
    print('Done')
# performence_test_with_imread()

def performence_test():
    img_list =[]
    for src in tqdm(src_list):
        img = imread(src)
        if img is not None:
            img_list.append(img)
    print('imread done! read %s images'%len(img_list))

    gauss, src_shape = deepgaze.predict(img_list[0])
    print('First blood %s'%str(src_shape))
    print('Initialize complated')

    for i, src in enumerate(tqdm(img_list)):
        gauss, src_shape = deepgaze.predict(src)
        y, x, h, w = deepgaze.crop(gauss, src_shape,(392, 698),True)   # (312, 698) 
        print(y,x,h,w)
    print('Done')
performence_test()