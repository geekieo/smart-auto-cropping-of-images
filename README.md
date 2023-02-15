# Smart Auto-Cropping of Images

## Abstract
The purpose of this project is to automatically crop the image to the target ratio based on visual attention.

## DeepGaze Body Architecture
Layer #|Kernel Size|Stride|Dilation|Padding|Input Channel|Output Channel|Input Size|Output Size|Receptive Field|name|remark|
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
1	|3         |1	|1	|2	|3  |64 |224 |224 |3   |conv1_1|╮
2	|3         |1	|1	|2	|64 |64 |224 |224 |5   |conv1_2|┃
3	|2         |2	|1	|0	|64 |64 |224 |112 |6   |pool1  |┃
4	|3         |1	|1	|2	|64 |128|112 |112 |10  |conv2_1|┃
5	|3         |1	|1	|2	|128|128|112 |112 |14  |conv2_2|┃
6	|2         |2	|1	|0	|128|128|112 |56  |16  |pool2  |┃
7	|3         |1	|1	|2	|128|256|56  |56  |24  |conv3_1|┃
8	|3         |1	|1	|2	|256|256|56  |56  |32  |conv3_2|┃
9	|3         |1	|1	|2	|256|256|56  |56  |40  |conv3_3|┣VGG16
10	|2         |2	|1	|0	|256|256|56	 |28  |44  |pool3  |┃
11	|3         |1	|1	|2	|256|512|28	 |28  |60  |conv4_1|┃
12	|3         |1	|1	|2	|512|512|28	 |28  |76  |conv4_2|┃
13	|3         |1	|1	|2	|512|512|28	 |28  |92  |conv4_3|┃
14	|2         |2	|1	|0	|512|512|28	 |14  |100 |pool4  |┃
15	|3         |1	|1	|2	|512|512|14	 |14  |132 |conv5_1|┃
16	|3         |1	|1	|2	|512|512|14	 |14  |164 |conv5_2|┃
17	|3         |1	|1	|2	|512|512|14	 |14  |196 |conv5_3|╯
18	|1         |1	|1	|2	|512|32 |14	 |14  |196 |conv1  |╮
19	|1         |1	|1	|2	|32 |16 |14	 |14  |196 |conv2  |┃
20	|1         |1	|1	|2	|16 |8  |14	 |14  |196 |conv3  |┣Readout
21	|1         |1	|1	|2	|8  |1  |14	 |14  |196 |conv4  |┃
22	|32        |16  |1  |2	|1  |1  |224 |224 |196 |deconv |╯
23	|153(fixed)|1   |1  |2  |1  |1  |224 |224 |196 |blur   |

## Reference

[1] Speedy Neural Networks for Smart Auto-Cropping of Images, https://blog.twitter.com/engineering/en_us/topics/infrastructure/2018/Smart-Auto-Cropping-of-Images.html  
[2] M. Kümmerer, L. Theis and M.Bethge. Deep Gaze I: Boosting Saliency Prediction with Feature Maps Trained on ImageNet. In *2015 International Conference on Learning Representations - Workshop Track(ICLR)*, May 2015  
**[3] M. Kümmerer, T.S.A. Wallis, M. Bethge. DeepGaze II: Reading fixations from deep features trained on object recognition. arXiv:1610.01563v1, 5 October 2016**  
[4] M. Kümmerer, T.S.A. Wallis, L.A. Gatys, M. Bethge. Understanding Low- and High-Level Contributions to Fixation Prediction. ICCV, 2017  
[5] L. Theis, I. Korshunova, A. Tejani, F. Huszar. Faster gaze prediction with dense networks and Fisher pruning. arXiv:1801.05787, 2018  
[6] J. Long, E. Shelhamer, T. Darrel. Fully Convolutional Networks for Semantic Segmentation. arXiv:14411.4038v2, 8 Mar 2015  
[7] Maoke Yang, Kun Yu, Chi Zhang, Zhiwei Li, Kuiyuan Yang. DenseASPP for Semantic Segmentation in Street Scenes. CVPR, 2018  
[8] A.G. Howard, M. Zhu, Bo Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, H. Adam. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861v1, 17 Apr 2017  
[9] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L. Chen. MobileNetV2: Inverted Residuals and Linear Bottlenecks. arXiv:1801.04381v3, 2 Apr 2018  


## Requirements

create and activate an environment based on **Python 3**  
    pip install opencv-python  
    pip install tqdm
    pip install tensorflow-gpu
