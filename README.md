# tensorflow-training-workflow




整个workflow是从tfrecords中读取，然后送入模型的方式。
注意这种数据读取方式需要
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


1.  model.py里面是模型的框架，输出可以softmax输出也可以是logits
2.  train.py需要调整的read_data_from_tfrecords.py中batch的个数
3.  predict.py需要去调整ckpt文件的位置




几个训练中的Bug:
1.训练时loss很大，但是准确率是对的。原因是img没有从0-255归一化到-0.5到0.5  tf.cast(img, tf.float32) * (1. / 255) - 0.5
2.之前loss不对，是因为tf.nn.softmax_cross_entropy_with_logits_v2用法错误，
tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y1)中的logits是模型的输出，不能是tf.nn.softmax的输出，
这个函数是对logits首先进行softmax。



