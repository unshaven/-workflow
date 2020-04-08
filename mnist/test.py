import tensorflow as tf

a = tf.Variable([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
b = tf.Variable([[0., 0., 0.9, 0.1, 0., 0., 0., 0., 0., 0.],
       [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
c = tf.Variable([1,2,3,4],dtype=tf.float32)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=b,labels=a)
    print sess.run(cross_entropy)
    loss = tf.reduce_sum(cross_entropy)
    loss1 = -tf.reduce_sum(a * tf.log(tf.nn.softmax(b)))
    print sess.run(loss)
    print sess.run(tf.nn.softmax(b))
    print sess.run(a * tf.log(tf.nn.softmax(b)))
    print sess.run(loss1)
    print sess.run(tf.nn.softmax())


