# import os
# import cv2  # 调用OpenCV
# import torch  # 调用PyTorch
# import tensorflow as tf  # 调用TensorFlow
# import numpy as np
# import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"  # https://blog.csdn.net/qq_45266796/article/details/109028605
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
# path_tools = os.path.join("tools","common_tools.py")
# print(path_tools)
# from tools.common_tools import transform_invert,set_seed
# set_seed(3)

class ImageProcessor:
    ''' This code is designed for the demonstration of 2D convolution processing of images. '''
    def __init__(self):
        self.name = ImageProcessor
        self.default_gallery_dir = os.path.dirname(__file__)+'/'+'Gallery'  # 设置默认图库目录

    def Conv2D(self,file_name='CUHK_demo.jpg',**kwargs):
        directory = kwargs['directory'] if 'directory' in kwargs else self.default_gallery_dir  # 指定图像文件所在目录

        image_file = directory+'/'+file_name  # 要处理图像的绝对地址

        return

if __name__ == '__main__':
    # 加载图像
    img = Image.open('D:/PycharmProjects/DeepLearning/Gallery/CUHK_demo_3.jpg').convert('RGB')  # 0~255
    imgL = img.convert('L')
    # plt.imshow(imgL)

    img_transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = img_transform(imgL)
    # 在dim=0维度增加一个批次
    # 因为卷积的输入tensor必须是四维的：
    # input输入：(N, C, H, W)
    # https://blog.csdn.net/qq_36793268/article/details/107375314
    # https://blog.csdn.net/qq_42178122/article/details/116453180
    img_tensor.unsqueeze_(dim=0)  # C*H*W to B*C*H*W

    print(img_tensor.shape)

    # 自定义卷积核
    kernel_dict = {
                    'Gaussian_Blur': torch.tensor([[[[1.0/16.0, 1.0/8.0, 1.0/16.0],
                                                        [1.0/8.0, 1.0/4.0, 1.0/8.0],
                                                        [1.0/16.0, 1.0/8.0, 1.0/16.0]]]]),
                    'Sharpen':  torch.tensor([[[[0.0,-1.0,0.0],
                                                [-1.0, 5.0, -1.0],
                                                [0.0, -1.0, 0.0]]]]),
                    'Prewitt_x': torch.tensor([[[[1.0, 1.0, 1.0],
                                                [0.0, 0.0, 0.0],
                                                [-1.0, -1.0, -1.0]]]]),
                    'Prewitt_y': torch.tensor([[[[-1.0, 0.0, 1.0],
                                                 [-1.0, 0.0, 1.0],
                                                 [-1.0, 0.0, 1.0]]]])
                    }

    mode = 'Prewitt_x'  # 卷积模式

    # 创建卷积层
    flag = 1
    # flag = 0
    if flag:
        conv_layer = nn.Conv2d(3, 1, 3, bias=False)  # input:(i, o, size) weights:(o, i , h, w)
        # nn.init.xavier_normal_(conv_layer.weight.data)  # Kernel参数初始化
        conv_layer.weight = nn.Parameter(kernel_dict[mode])  # 自定义Kernel
        print(conv_layer.weight.data)

        img_conv = conv_layer(img_tensor)

    transform_invert = transforms.Compose([transforms.ToPILImage()])

    # 可视化
    print("卷积前尺寸：{}\n卷积后尺寸：{}".format(img_tensor.shape, img_conv.shape))
    #img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
    #img_raw = transform_invert(img_tensor.squeeze(), img_transform)
    img_conv = transform_invert(img_conv.squeeze())
    img_raw = transform_invert(img_tensor.squeeze())
    # plt.subplot(122).imshow(img_conv, cmap='gray')
    #plt.subplot(121).imshow(img_raw)
    #plt.subplot(122).imshow(img_conv)
    plt.imshow(img_conv)
    # plt.imshow(imgL)
    plt.axis('off')  # 去掉坐标轴

    plt.savefig('D:/PycharmProjects/DeepLearning/Gallery/Conv2D/'+mode+'.eps', dpi=600, format='eps')

    plt.show()




    # print(image.shape)
    # 加载图像
    #img = Image.open('lena.png').convert('RGB')  # 0~255
    #plt.imshow(img)


# OpenCV
#if __name__ == '__main__':
    #image = cv2.imread('D:/PycharmProjects/DeepLearning/CUHK_demo.jpg')
    #image = cv2.resize(image, (200, 200))
    #image = np.expand_dims(image, 0).reshape((1, 200, 200, 3))

    #print(image.shape)

    #input_tensor = tf.placeholder(tf.float32, shape=(1, 200, 200, 3))
    #weights = tf.get_variable("weights", shape=[3, 3, 3, 4],
                          #initializer=tf.random_normal_initializer())
    #conv1 = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding="SAME")

    #sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    #output_tensor = sess.run(conv1, feed_dict={input_tensor: image})

    #for i in range(4):
        #plt.subplot(2, 2, i+1)
        #plt.imshow(output_tensor[0][:, :, i])
    #plt.show()
