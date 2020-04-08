#coding:utf-8
import tensorflow as tf
import model
from read_from_tfrecords import read_data_from_tfrecords
import time
import os

dir_name = '/Users/wuji/Documents/study/pycharm/mnist/dataset/images/MnistTfrecords'
filenames = os.listdir(dir_name)
#train
n_classes = 10
def train():
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,[None,28,28,1],name='x')
        y1 = tf.placeholder(tf.float32,[None,n_classes],name='y')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    with tf.name_scope('output'):
        y_,logits,[W,b] = model.regression(x)
        # y_,logits,test_variable = model.convolutional(x,keep_prob)
    # tf.reset_default_graph()

    # init = tf.group(tf.global_variables_initializer(),
    #                 tf.local_variables_initializer())
    # sess.run(init)
    # print sess.run([y,y_],feed_dict={x:b_x,y:b_y})
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y1)
    loss = tf.reduce_sum(cross_entropy)
    # loss1 = -tf.reduce_sum(y1 * tf.log(y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y1,1),tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')

    imgs, y = read_data_from_tfrecords(filenames)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    # tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 1
        for _ in range(10000):
            b_x,b_y = sess.run([imgs,y])
            start = time.time()
            # print b_y
            # print "y和y_"
            # val = tf.nn.softmax(test_variable[-1])
            # W_conv1 = test_variable[0]
            # print sess.run(test_variable[-1],feed_dict={x:b_x,y1:b_y})
            # print sess.run(val,feed_dict={x:b_x,y1:b_y})
            # print sess.run([y1,y_],feed_dict={x:b_x,y1:b_y})
            # print sess.run(y1 * tf.log(y_),feed_dict={x:b_x,y1:b_y})
            # print sess.run(cross_entropy,feed_dict={x:b_x,y1:b_y})
            # print sess.run(loss,feed_dict={x:b_x,y1:b_y})
            # print sess.run(loss1,feed_dict={x:b_x,y1:b_y})
            # print sess.run(loss1,feed_dict={x:b_x,y1:b_y,keep_prob:0.5})
            sess.run([train_step],feed_dict={x:b_x,y1:b_y,keep_prob:0.5})
            # _,loss,auc = sess.run([train_step,loss,accuracy],feed_dict={x:b_x,y1:b_y})
            if step % 100 == 0:
                l,auc = sess.run([loss,accuracy],feed_dict={x:b_x,y1:b_y,keep_prob:1})
                print("iter "+str(step)+', minibatch loss= {}'.format(l) + ' training accuracy = {}'.format(round(auc,4)))
            # if step % 10 == 0:
            #     print('iter %d, duration: %.2fs' %(step,time.time()-start))
            step += 1
        saver.save(sess,'./mnist_model_regression.ckpt')
        coord.request_stop()
        coord.join(threads=threads)
        print("finiash training！")



if __name__ == '__main__':
    # imgs, y = read_data_from_tfrecords(filename)
    #
    # with tf.Session() as sess:
    #     init = tf.group(tf.global_variables_initializer(),
    #                     tf.local_variables_initializer())
    #     sess.run(init)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     b_x, b_y = sess.run([imgs, y])
    #
    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train()