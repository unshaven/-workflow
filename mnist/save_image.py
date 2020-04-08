#coding=utf-8
import numpy as np
import cv2
import os
print os.path.abspath(__file__)
def check_mnist_image_label():
    with open('../mnist/dataset/train-images-idx3-ubyte','rb') as f:
        f1 = f.read()
        print '图片数目为{}'.format(int(f1[4:8].encode('hex'),16))
        image1 = [int(item.encode('hex'),16) for item in f1[16:16+28*28]]
        image_np = np.array(image1,dtype=np.uint8).reshape(28,28,1)
    with open('../mnist/dataset/train-labels-idx1-ubyte','rb') as f:
        f1 = f.read()
        label1 = [int(item.encode('hex'),16) for item in f1[8]]
        print label1

def save_mnist_to_jpg(mnist_image_file,mnist_label_file,save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if 'train' in os.path.basename(mnist_image_file):
        num_file = 60000
        prefix = 'train'
    else:
        num_file = 10000
        prefix = 'test'
    with open(mnist_image_file) as f:
        image_file = f.read()
    with open(mnist_label_file) as f:
        label_file = f.read()
    image_file = image_file[16:]
    label_file = label_file[8:]
    for i in range(num_file):
        image = [int(item.encode('hex'),16) for item in image_file[i*784:i*784+784]]
        label = [int(label_file[i].encode('hex'),16)]
        image_np = np.array(image,dtype=np.uint8).reshape(28,28,1)
        save_name = os.path.join(save_dir,'{}_{}_{}.jpg'.format(prefix,i,label))
        cv2.imwrite(save_name,image_np)
        if i % 1000 == 0:
            print '{} has processed'.format(i)
if __name__ == '__main__':
    # mnist_image_file = 'dataset/train-images-idx3-ubyte'
    # mnist_label_file = 'dataset/train-labels-idx1-ubyte'
    save_dir = 'dataset/mnist_test_images'
    mnist_image_file = 'dataset/t10k-images-idx3-ubyte'
    mnist_label_file = 'dataset/t10k-labels-idx1-ubyte'
    save_mnist_to_jpg(mnist_image_file,mnist_label_file,save_dir)