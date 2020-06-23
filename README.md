# Tic-Tac-Toe
Tic-Tac-Toe Keras, TensorFlow Programming

Keras, TensorFlow. 이하 두 가지의 API를 활용하여 진행했습니다.

 Keras는 Python으로 작성되었으며 TensorFlow, CNTK 또는 Theano 위에서 실행할 수 있는 고급 신경망 API입니다. 빠른 실험을 가능하게 하는 데 중점을 두고 개발되었습니다.
 
 TensorFlow는 머신러닝을 위한 엔드 투 엔드 오픈소스 플랫폼입니다. 도구, 라이브러리, 커뮤니티 리소스로 구성된 포괄적이고 유연한 생태계를 통해 연구원들은 ML에서 첨단 기술을 구현할 수 있고 개발자들은 ML이 접목된 애플리케이션을 손쉽게 빌드 및 배포할 수 있습니다. TensorFlow 2.0은 단순성과 편의성에 초점을 두며, Keras와 즉시 실행(eager execution)을 이용한 쉬운 모델 작성을 지원하게 되었습니다.
 
 주어진 Tic-Tac-Toe 데이터 세트와 Keras, TensorFlow를 활용하여 데이터를 학습하는 과정을 살펴보고, 학습방법을 여러 가지 방향으로 변경한 후 나온 결과를 검증하여 최적의 학습모델을 찾아보고자 하였습니다.

UCI Machine Learning Repository에서 제공하는 Tic-Tac-Toe Endgame Data Set을 학습을 진행하기 위하여 다음과 같이 수치화 하여 진행하였습니다.
(https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)


|tic-tac-toe.csv|x|o|b|true|false|
|---|---|---|---|---|---|
|data.csv|-1|1|0|1, 0|0, 1|

입력받은 데이터 세트 학습을 위해 4층의 구조를 가진 Multi-Layer Perceptron을 구성하였습니다. 구성한 MLP 구조는 2개의 은닉층은 relu를, 출력층은 softmax와 sigmoid를 사용하여 구성하였습니다.

(Graph)

 1, 2, 3) 그래프를 살펴보면 옵티마이저를 Adam을 사용하였을 때 SGD보다 급격한 loss와 accuracy 값 상향이 발생하였고, 반면 RMSProp를 사용하였을 때는 loss와 accuracy 값이 epoch 초기에 엎치락뒤치락하는 상황이 발생하지만, epoch가 늘어날수록 안정화된 학습률을 나타내고 있습니다.
 4, 5) 출력 레이어를 softmax를 사용할 시 sigmoid보다 약간 더 나은 학습률을 보여주고 있습니다.
 6, 7, 8) 학습률을 각 0.01, 0.5, 1로 진행하였을 시 초기 0.01일 때 가장 나은 학습을 보여주며, 0.5는 비교적 평이함을 보였고, 1일시 학습 초기에 accuracy가 증가하다 급격하게 떨어지는 모습을 보입니다.
 9, 10) epoch를 절반가량 횟수를 줄이고 늘렸을 때는 base 학습방법과 큰 차이를 보이지 않고 있습니다. 하지만 그래도 epoch가 많이 진행된 학습이 더 나은 결과를 보여주고 있습니다. 이러한 차이는 한정된 데이터 세트를 사용하여 학습하기에 나오는 결과로 예측됩니다. 더 많은 데이터를 사용하는 학습 일 시에는 많은 차이를 보일 것으로 예상합니다.
