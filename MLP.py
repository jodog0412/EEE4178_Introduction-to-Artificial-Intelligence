# coding: utf-8

# 인공지능(딥러닝)개론 # Homework 1
# 간단한 XOR Table을 학습하는 NN을 구성하는 문제입니다.
# 
#  1-Layer, 2-Layer model을 각각 구성하여 XOR 결과를 비교합니다.
#  1-Layer, 2-Layer의 model을 feedforward network와 Backpropagation을 이용하여 학습시킵니다.
#  주어진 양식을 활용해 주시며, scale, 차원의 순서, hyper parameter등은 결과가 잘 나오는 방향으로 Tuning하셔도 무방합니다.
#  Layer의 Activation 함수인 Tanh는 54~57번째 줄의 함수를 사용하시면 됩니다.
#  결과 재현을 위해 Weight, bias 값을 저장하여 함께 첨부해 주시기 바랍니다.
#  각 모델에서 loss 그래프와 testing step을 첨부하여 간단하게 자유 양식 결과 보고서(2~3장 내외)로 작성해 주세요.
# 
# 
#  양식에서 활용하는 라이브러리 외에 추가로 import 하여 사용하실 수 없습니다.



## 이 외에 추가 라이브러리 사용 금지
import numpy as np
import random
import matplotlib.pyplot as plt



# Hyper parameters
## 학습의 횟수와 Gradient update에 쓰이는 learning rate입니다.
## 다른 값을 사용하여도 무방합니다.
epochs = 10000
learning_rate = 0.0001



# Input data setting
## XOR data 
## 입력 데이터들, XOR Table에 맞게 정의해놓았습니다.
train_inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_out = np.array([0, 1, 1, 0])



# Weight Setting
## 학습에 사용되는 weight들의 초기값을 선언해 줍니다. random이 아닌 다른 값을 사용하여도 무방합니다.
## 현재 weight변수는 2-layer 기준으로 설정되어있습니다.
## 1-layer의 경우 W1과 b1을 문제에 맞게 바꿔 진행해주시면 됩니다. (방법은 자유)

# 1-layer
W1 = np.random.randn(2,1)
b1 = np.random.randn(1,1)

# 2-layer
# W1 = np.random.randn(2,2)
# W2 = np.random.randn(2,1)
# b1 = np.random.randn(1,2)
# b2 = np.random.randn(1,1)


##-----------------------------------##
##------- Activation Function -------##
##-----------------------------------##
def tanh(x):
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator/denominator


# ----------------------------------- #
# --------- Training Step ----------- #
# ----------------------------------- #
# 학습이 시작됩니다.
# epoch 사이즈만큼 for 문을 돌며 학습됩니다.
# 빈 칸을 채워 Weight과 bias를 학습(update)하는 신경망을 설계하세요.
# 양식의 모든 내용을 무조건 따를 필요는 없습니다. 각자에게 편하게 수정하셔도 좋습니다. (변경한 경우 보고서에 작성 부탁드립니다.)

errors = []
for epoch in range(epochs):
        
    # 데이터 4가지 중 랜덤으로 하나 선택
    for batch in range(4):
        idx = random.randint(0,3)

        # 입력 데이터 xin과 해당하는 정답 ans 불러오기
        xin = train_inp[idx].reshape(1,2)
        ans = train_out[idx]
        
        # 1-layer
        # Layer에 맞는 Forward Network 구성
        net1=xin@W1+b1
        z=tanh(net1)
        
        # Mean Squared Error (MSE)로 loss 계산
        loss = (ans-z)**2
        
        # delta matrix initialization(Zero 값이 아닌 다른 방법으로 이용하셔도 됩니다.)
        delta_W1=np.zeros((2,1))
        delta_b1=np.zeros((1,1))

        # Backpropagation을 통한 Weight의 Gradient calculation(update)
        z_grad=1-z**2
        delta_W1=2*(z-ans)*z_grad*(xin.T)
        delta_b1=0

        # 각 weight의 update 반영
        W1 = W1 - learning_rate * delta_W1
        b1 = b1 - learning_rate * delta_b1

        # # 2-layer
        # # Layer에 맞는 Forward Network 구성
        # net1 = xin@W1+b1
        # y=tanh(net1)
        # net2 = y@W2+b2
        # z=tanh(net2)

        # # Mean Squared Error (MSE)로 loss 계산
        # loss = (ans-z)**2

        # # delta matrix initialization(Zero 값이 아닌 다른 방법으로 이용하셔도 됩니다.)
        # delta_W1 = np.zeros((2,2))
        # delta_W2 = np.zeros((2,1))
        # delta_b1 = np.zeros((1,2))
        # delta_b2 = np.zeros((1,1))
        
        # # Backpropagation을 통한 Weight의 Gradient calculation(update)
        # y_grad=1-y**2
        # z_grad=1-z**2
        # delta_W2 = 2*(z-ans)*z_grad*(y.T)
        # delta_W1 = 2*(z-ans)*z_grad*np.sum(W2)*xin.T@y_grad
        # delta_b1 = 0
        # delta_b2 = 0

        # # 각 weight의 update 반영
        # W1 = W1 - learning_rate * delta_W1
        # W2 = W2 - learning_rate * delta_W2
        # b1 = b1 - learning_rate * delta_b1
        # b2 = b2 - learning_rate * delta_b2
        
        
    ## 500번째 epoch마다 loss를 프린트 합니다.]
    if epoch%500 == 0:
        print("epoch[{}/{}] loss: {}".format(epoch,epochs,loss.astype(np.float64)))
        
    ## plot을 위해 값 저장
    errors.append(loss)



## 학습이 끝난 후, loss를 확인합니다.
loss =  np.array(errors)
# print("W1:{},xin:{},W2:{},y:{},z:{},b1:{},b2:{}".format(W1.shape,xin.shape,W2.shape,y.shape,z.shape,b1.shape,b2.shape))
plt.plot(range(epochs),loss.reshape(10000,-1))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()



#-----------------------------------#
#--------- Testing Step ------------#
#-----------------------------------#

for idx in range(4):
    xin = train_inp[idx]
    ans = train_out[idx]
    
    # 1-layer
    net1=xin@W1+b1
    z=tanh(net1)

    # 2-layer
    # net1 = xin@(W1)+b1
    # y=tanh(net1)
    # net2 = y@(W2)+b2
    # z=tanh(net2)

    pred = z # ans와 가까울 수록 잘 학습된 것을 의미합니다.
    
    print("input: ", xin, ", answer: ", ans, ", pred: {}".format(pred.astype(float)))
    


#-----------------------------------#
#--------- Weight Saving -----------#
#-----------------------------------#

# weight, bias를 저장하는 부분입니다.
# 학번에 자신의 학번으로 대체해 주세요.

    #layer 1개인 경우
    # np.savetxt("20181342_layer1_weight",(W1, b1),fmt="%s")

    #layer 2개인 경우
    # np.savetxt("20181342_layer2_weight",(W1, W2, b1, b2),fmt="%s")

