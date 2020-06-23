# 사용 패키지
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# 데이터셋 생성
dataset = np.genfromtxt('tic-tac-toe.csv', delimiter=',', dtype=np.float32)

x_train = dataset[:576, 0:9]
y_train = dataset[:576, [1]]
x_val = dataset[382:, 0:9]
y_val = dataset[382:, [1]]

# 훈련셋, 검증셋 선택
train_rand_idxs = np.random.choice(576, 64)
val_rand_idxs = np.random.choice(382, 64)
x_train = x_train[train_rand_idxs]
y_train = y_train[train_rand_idxs]
x_val = x_val[val_rand_idxs]
y_val = y_val[val_rand_idxs]

# 라벨링 전환
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

# 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=9, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# 모델 학습과정 설정
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
hist = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))

# 학습과정 가시화
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

# 6. 학습결과 도출
loss_and_metrics = model.evaluate(x_val, y_val, batch_size=64)
print('result')
print('loss : ' + str(loss_and_metrics[0]))
print('accuracy : ' + str(loss_and_metrics[1]))
plt.show()
