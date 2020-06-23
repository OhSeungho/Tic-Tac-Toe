from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

def inference(x, keep_prob, n_in, n_hiddens, n_out):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    # 입력층-은닉층과 은닉층-은닉층 설계
    for i, n_hidden in enumerate(n_hiddens):
        if i == 0:  # 입력층-은닉층
            input = x
            input_dim = n_in
        else:   # 은닉층-은닉층
            input = output  # 출력층이 다음 입력층이 된다
            input_dim = n_hiddens[i-1]

        W = weight_variable([input_dim, n_hidden])  # weight의 shape은 입력층 차원(784) x 뉴런의 개수
        b = bias_variable([n_hidden])               # bias의 shape은 뉴런의 개수

        hidden_layer = tf.nn.relu(tf.matmul(input, W) + b)  # 은닉층의 활성화 함수로 ReLU를 사용
        output = tf.nn.dropout(hidden_layer, keep_prob=keep_prob)   # 출력은 드롭아웃을 적용

    # 은닉층-출력층 설계
    W_out = weight_variable([n_hiddens[-1], n_out]) # 출력층의 weight shape은 뉴런의 개수 x 출력층의 차원(10개)
    b_out = bias_variable([n_out])                  # bias의 shape은 출력층의 차원

    hypothesis = tf.nn.softmax(tf.matmul(output, W_out) + b_out)    # 출력 활성화 함수는 softmax를 사용

    return hypothesis

def loss(hypothesis, y):
    # cross entropy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), reduction_indices=[1]))

    return cross_entropy

def train(loss):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    return train

def accuracy(hypothesis, y):
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


if __name__ == '__main__':
    MNIST = datasets.fetch_mldata('MNIST original', data_home='.')

    n = len(MNIST.data)
    N = int(n * 1.0)
    indices = np.random.permutation(range(n))[:N]
    X = MNIST.data[indices]
    Y = MNIST.target[indices]
    Y = np.eye(10)[Y.astype(int)]  # 1-of-K hot coding 표현으로 변환

    test_size = 0.2
    # 학습 데이터와 테스트 데이터를 80:20으로 나눈다
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # 학습 데이터의 20%를 검증 데이터로 사용한다
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=test_size)

    # 2. 모델을 설정한다
    # 입출력 데이터와 은닉층의 뉴런 개수 설정
    n_in = len(X[0])  # 입력 데이터 이미지의 크기 28x28(=784)
    n_hiddens = [200, 200, 200]  # 각 은닉층의 뉴런 개수
    n_out = len(Y[0])  # 출력 데이터의 개수 (0~9)
    p_keep = 0.5        # 드롭아웃 확률의 비율

    x = tf.placeholder(tf.float32, shape=[None, n_in])
    y = tf.placeholder(tf.float32, shape=[None, n_out])

    keep_prob = tf.placeholder(tf.float32)

    # 가설식을 그래프에 추가
    hypothesis = inference(x, keep_prob, n_in=n_in, n_hiddens=n_hiddens, n_out=n_out)

    # 오차 함수를 그래프에 추가
    loss = loss(hypothesis, y)

    # 학습 함수를 그래프에 추가
    train = train(loss)

    # 정확도 예측 함수 그래프에 추가
    accuracy = accuracy(hypothesis, y)

    # 3. 모델을 학습시키면서 그래프를 실행
    epochs = 50
    batch_size = 200

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    n_batches = len(X_train) // batch_size

    # 학습 진행 상황 데이터를 저장할 변수 설정
    history = {'val_loss': [],
               'val_acc': []}

    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)

        for i in range(n_batches):

            start = i * batch_size
            end = start + batch_size

            sess.run(train, feed_dict={x: X_[start: end], y: Y_[start: end], keep_prob: p_keep})

        # 검증 데이터를 사용해 정확도 측정
        val_loss = loss.eval(session=sess, feed_dict={x: X_val,
                                                      y: Y_val,
                                                      keep_prob: 1.0})

        val_accuracy = accuracy.eval(session=sess, feed_dict={x: X_val,
                                                              y: Y_val,
                                                              keep_prob: 1.0})

        # 검증 데이터에 대한 학습 진행 상황을 저장
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy * 100)

        print(f'epoch : {epoch:03}, Validation loss : {val_loss:.7}, Validation accuracy : {val_accuracy*100:.6} %')

    # 4. 모델을 평가한다
    accuracy_rate = accuracy.eval(session=sess, feed_dict={x: X_test, y: Y_test, keep_prob: 1.0})
    print(f'Test Accuracy : {accuracy_rate*100:.6} %')

    # 폰트 설정
    font_name = matplotlib.font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    matplotlib.rc('font', family=font_name)
    # 도화지 생성
    fig = plt.figure()
    # 정확도 그래프 그리기
    plt.plot(range(epochs), history['val_acc'], label='Accuracy', color='darkred')
    # 축 이름
    plt.xlabel('epochs')
    plt.ylabel('검증 정확도(%)')
    plt.title('epoch에 따른 검증 데이터에 대한 정확도 그래프')
    plt.grid(linestyle='--', color='lavender')
    # 그래프 표시
    plt.show()
