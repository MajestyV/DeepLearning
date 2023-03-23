import os
import cv2  # 调用OpenCV
import tensorflow as tf  # 调用TensorFlow
import numpy as np
import matplotlib.pyplot as plt

class conv2d:
    ''' This code is designed for the demonstration of 2D convolution processing of images. '''
    def __init__(self):
        self.name = conv2d
        self.default_dir =

if __name__ == '__main__':
    image = cv2.imread("/Users/jmc/Desktop/gecko.jpg")
    image = cv2.resize(image, (200, 200))
    image = np.expand_dims(image, 0).reshape((1, 200, 200, 3))

    print(image.shape)

    input_tensor = tf.placeholder(tf.float32, shape=(1, 200, 200, 3))
    weights = tf.get_variable("weights", shape=[3, 3, 3, 4],
                          initializer=tf.random_normal_initializer())
    conv1 = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding="SAME")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output_tensor = sess.run(conv1, feed_dict={input_tensor: image})

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(output_tensor[0][:, :, i])
    plt.show()
