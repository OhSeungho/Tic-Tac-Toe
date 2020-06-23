import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.set_random_seed(777)

# dataset
dataset = np.genfromtxt('data.csv', delimiter=',', dtype=np.float32)
np.random.shuffle(dataset)

x_data = dataset[:, 0:-2]
y_data = dataset[:, 9:]

N_TRAIN = 576
N_TEST = 383

x_train = x_data[:N_TRAIN]
y_train = y_data[:N_TRAIN]

x_test = x_data[N_TRAIN:]
y_test = y_data[N_TRAIN:]

X = tf.compat.v1.placeholder(tf.float32, [None, 9])
Y = tf.compat.v1.placeholder(tf.float32, [None, 2])

# layer 1
W1 = tf.Variable(tf.random.uniform([9, 100], -0.5, 0.5))
b1 = tf.Variable(tf.random.uniform([100], -0.5, 0.5))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

# layer 2
W2 = tf.Variable(tf.random.uniform([100, 50], -0.5, 0.5))
b2 = tf.Variable(tf.random.uniform([50], -0.5, 0.5))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

# out layer
W3 = tf.Variable(tf.random.uniform([50, 2], -0.5, 0.5))
b3 = tf.Variable(tf.random.uniform([2], -0.5, 0.5))
hypothesis = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)

# parameters
epochs = 1000
learning_rate = 0.01

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#cost = tf.reduce_mean(-1 * tf.reduce_sum(Y * tf.log(hypothesis), axis=1))


# prediction
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


 # optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# session
List_accuracy = []
List_loss = []
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# train
for epoch in range(epochs):
    feed_dict = { X: x_train, Y: y_train }
    c, _, a = sess.run([cost, optimizer, accuracy], feed_dict=feed_dict)
    print('Epoch:', '%04d' % (epoch + 1), 'Cost =', '{:.9f}'.format(c), "Accuracy: ", a)
    List_accuracy.append(a)
    List_loss.append(c)

# match
for i in range(N_TEST-1):    
    feed_dict = { X: [x_test[i]]}
    result = sess.run(hypothesis, feed_dict = feed_dict)
#   print(result, y_test[i])
    

# graph
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
acc_ax.plot(List_accuracy, 'r', label="accuracy")
loss_ax.plot(List_loss, 'b', label="loss")
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()
