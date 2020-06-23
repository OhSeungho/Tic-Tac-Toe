
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(777)  # for reproducibility


xy = np.genfromtxt('data.csv', delimiter=',', dtype=np.float32)
np.random.shuffle(xy)

x_data = xy[:, 0:-2]
y_data = xy[:, 9:]

train_num = 576
test_num = 383

train_x = x_data[: train_num]
train_y = y_data[: train_num]

test_x = x_data[train_num:]
test_y = y_data[train_num:]

X = tf.compat.v1.placeholder(tf.float32, [None, 9])
Y = tf.compat.v1.placeholder(tf.float32, [None, 2])

# W = tf.Variable(tf.random.normal([9, 1]))
# B = tf.Variable(tf.random.normal([1]))
# HH = tf.matmul(X, W) + B

# 1층 노드 100개
W1 = tf.Variable(tf.random.uniform([9, 100], -0.5, 0.5))
b1 = tf.Variable(tf.random.uniform([100], -0.5, 0.5))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

# 2층 노드 50개
W2 = tf.Variable(tf.random.uniform([100, 50], -0.5, 0.5))
b2 = tf.Variable(tf.random.uniform([50], -0.5, 0.5))
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# 출력층 1, 2
W3 = tf.Variable(tf.random.uniform([50, 2], -0.5, 0.5))
b3 = tf.Variable(tf.random.uniform([2], -0.5, 0.5))
hypothesis = tf.matmul(layer2, W3) + b3

# parameters
learning_rate = 0.05
training_epochs = 1000

# 손실
#cost = tf.reduce_mean(tf.square(hypothesis - Y))
#cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
#cost = tf.reduce_mean(-1 * tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost = tf.compat.v1.losses.mean_squared_error(hypothesis, Y)

# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits_y_pred)
# loss = tf.reduce_mean(loss)

# 최적화
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
#train = optimizer.minimize(cost)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, -1), tf.argmax(Y, -1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs_index = []
train_acc = []
test_acc = []
# train my mode
for epoch in range(training_epochs + 1):
    epochs_index.append(epoch)
    feed_dict = {X: train_x, Y: train_y}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

    for i in range(test_num-1):
        feed_dict = {X: [test_x[i]]}
    result = sess.run(hypothesis, feed_dict=feed_dict)
    # print(result, test_y[i])

    train_acc.append(sess.run([accuracy], feed_dict = { X: train_x, Y: train_y}))
    test_acc.append(sess.run([accuracy], feed_dict = { X: test_x, Y: test_y}))
    
    if ((epoch) % 200 == 0):
        #print('Epoch : %04d' % epoch, 'cost = %.9f' % c, 'train_acc = %.9f' % train_acc, 'test_acc = %.9f' % test_acc)
        print('Epoch : %04d' % epoch, 'cost = %.9f' % c)

print('Learning Finished!')

for a in range(len(epochs_index)):
    print(a, train_acc[a])



plt.plot(epochs_index, train_acc)
plt.plot(epochs_index, test_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()
