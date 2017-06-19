import tensorflow.examples.tutorials.mnist.input_data as id
import tensorflow as tf

mnist = id.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,w) + b)
y_ = tf.placeholder(tf.float32, [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    for i in range(1000) :
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

        #sess.run(train_step, feed_dict={x:mnist.train.images, y_:mnist.train.labels})
        print i,batch_xs.shape

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
