import io
import tensorflow as tf
import numpy as np

iris_train = []
iris_target = []
iris_target_dict = {'setosa':0,'versicolor':1,'virginica':2}
iris = io.open("./iris.csv")
i = 0

for line in iris :
    tmp_ls = [0 for i in range(4)]
    tmp_dict = [0,0,0]
    line = line.replace("\n","")
    tmp_ls[0],tmp_ls[1],tmp_ls[2],tmp_ls[3],tmp = line.split(',')
    for i in range(4) :
        tmp_ls[i] = float(tmp_ls[i])
    iris_train.append(tmp_ls)
    tmp_dict[iris_target_dict[tmp]] = 1
    iris_target.append(tmp_dict)
    i += 1
    #print line.split(',')

iris.close()

iris_testx = []
iris_testy = []
iris = io.open("./iris_test.txt")
i = 0

for line in iris :
    tmp_ls = [0 for i in range(4)]
    tmp_dict = [0,0,0]
    line = line.replace("\n","")
    tmp_ls[0],tmp_ls[1],tmp_ls[2],tmp_ls[3],tmp = line.split(',')
    for i in range(4) :
        tmp_ls[i] = float(tmp_ls[i])
    iris_testx.append(tmp_ls)
    tmp_dict[iris_target_dict[tmp]] = 1
    iris_testy.append(tmp_dict)
    i += 1
    #print line.split(',')

iris.close()
data_len = iris_target.__len__()

x = tf.placeholder(tf.float32,[None,4])
w = tf.Variable(tf.zeros([4,3]))
b = tf.Variable(tf.zeros([3]))
y = tf.nn.softmax(tf.matmul(x,w)+b)
y_ = tf.placeholder(tf.float32,[None,3])

iris_train = np.array(iris_train)
iris_target = np.array(iris_target)
iris_testx = np.array(iris_testx)
iris_testy = np.array(iris_testy)
print iris_testy.shape

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    for i in range(50000):
        sess.run(train_step,feed_dict = {x:iris_train,y_:iris_target})
        if i%100 == 0:
            print i

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print sess.run(accuracy,feed_dict = {x:iris_testx,y_:iris_testy})
