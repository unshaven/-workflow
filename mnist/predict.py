#coding:utf-8
import tensorflow as tf
import model
from read_from_tfrecords import read_data_from_tfrecords
import time
import numpy as np
import os
import cv2
from read_from_tfrecords import read_data_from_tfrecords

dir_name = '/Users/wuji/Documents/study/pycharm/mnist/dataset/TestMnistTfrecords'
filenames = os.listdir(dir_name)
# dir_name = '/Users/wuji/Documents/study/pycharm/mnist/dataset/images/MnistTfrecords'
# filenames = os.listdir(dir_name)
#save_file = '/Users/wuji/Documents/study/pycharm/mnist/mnist_model.ckpt.data-00000-of-00001'
save_file = '/Users/wuji/Documents/study/pycharm/mnist/mnist_model_regression.ckpt.data-00000-of-00001'
saver = tf.train.import_meta_graph('/Users/wuji/Documents/study/pycharm/mnist/mnist_model_regression.ckpt.meta')

imgs, y = read_data_from_tfrecords(filenames)
init = tf.global_variables_initializer()

def one_hot_to_normal(label):
    for l in range(len(label)):
        if label[l] == 1:
            return l
with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver.restore(sess,'/Users/wuji/Documents/study/pycharm/mnist/mnist_model_regression.ckpt')
    graph = tf.get_default_graph()
    xs = graph.get_tensor_by_name('input/x:0')
    ys = graph.get_tensor_by_name('input/y:0')
    keep_prob = graph.get_tensor_by_name('input/keep_prob:0')
    x_test,y_test = [],[]
    y_label = []
    for step in range(10000):
        if step % 100 == 0:
            print("iter : {}".format(step))
        images, labels = sess.run([imgs, y])
        # image_np = np.array(images,dtype=np.uint8).reshape(28,28,1)
        x_test.append(images)
        y_test.append(labels)
        y_label.append(one_hot_to_normal(labels))
        if step < 10:
            cv2.imwrite('test/test_regression_{}_{}.jpg'.format(step,one_hot_to_normal(labels)),(images+0.5)*255)
    # print y_label[:100],len(y_label)
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
    print sess.run(accuracy,feed_dict={xs:x_test,ys:y_test,keep_prob:1.0})
    coord.request_stop()
    coord.join(threads=threads)
    print("finiash training！")