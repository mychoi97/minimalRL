# TD Actor-Critic
# REINFORCE와 다른 부분만 주석을 통해 설명하겠다.
'''
이 코드에서 재미있는 부분:
loss = torch.cat(self.loss_list).sum()
이렇게 loss를 모아서 처리해서 batch 처리로 하는데, TD Actor-Critic은
매 step마다 update할 수 있다고 했었다. 근데 이렇게 batch로 처리하면 learning rate가 훨씬 커져도 학습이 잘 된다.
그리고 매 step 마다 update하는 것 보다 batch로 처리하면 훨씬 학습이 안정적이다.

loss = loss/len(self.loss_list)
loss를 self.loss_list의 length로 나눠주는데, 이걸 나눠주지 않아도 학습은 된다. 그런데, 나눠주면 학습이 더 잘된다.
'''
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# actor-critic 클래스
class ActorCritic(nn.Module):
    def __init__(self):
        
        # nn.Module __init__ 메서드 호출
        super(ActorCritic, self).__init__()
        
        # loss들을 담아놓을 리스트 생성
        self.loss_list = []

        # obs의 차원이 4개 이므로, input으로 4개를 받아서 output으로 128개를 내놓은 fully connected layer
        self.fc1 = nn.Linear(4, 128)
        
        # policy head와 value head 이렇게 두개로 갈라진다.
        # 128개의 input을 받아서 action(왼쪽, 오른쪽) 두개 값을 내놓는 fully connected layer
        self.fc_pi = nn.Linear(128, 2)

        # 128개의 input을 받아서 value 값을 내놓은 fully connected layer
        self.fc_v = nn.Linear(128, 1)

        # 최적화함수 Adam, learning rate 0.005
        self.optimizer = optim.Adam(self.parameters(), lr = 0.005)
    
    # 신경망 forward 과정
    def forward(self, x):
        # self.fc1을 통해서 나온 128차원의 값을 relu activation function을 거치도록 한다.
        x = F.relu(self.fc1(x))

         # pi는 확률 분포니까 self.fc_pi를 거쳐서 2차원의 값을 내놓고, softmax activation function까지 거쳐서 해당 action을 할 확률을 내놓도록 한다.
        pol = self.fc_pi(x)
        pi = F.softmax(pol, dim=0)

        # value는 어떤 값이든 가질 수 있으니까 그냥 1차원의 값을 내놓도록 self.fc_v를 거치도록 한다.
        v = self.fc_v(x)

        return pi, v

    def gather_loss(self, loss):
        # loss를 받아서 append해주는데, 그냥 append해주는게 아니고 pytorch의 함수인 unsqueeze를 이용해서 append해준다.
        # unsqueeze는 차원을 하나 늘려준다. 이걸 왜 하냐면 batch 처리 하려고 쓰는 것이다.
        # 나중에 loss들이 다 모여있으면 train 할때, loss list들을 concatenate, 즉 다 붙여준다. 이걸 하기위해서는 unsqueeze를 사용해야한다.
        self.loss_list.append(loss.unsqueeze(0))

    # train
    def train(self):
        # loss list들을 concatenate해서 다 더해준다.
        # 이렇게 loss를 모아서 처리해서 batch 처리로 하는데, TD Actor-Critic은 매 step마다 update할 수 있다고 했었다. 
        # 근데 이렇게 batch로 처리하면 learning rate가 훨씬 커져도 학습이 잘 된다.
        # 그리고 매 step 마다 update하는 것 보다 batch로 처리하면 훨씬 학습이 안정적이다.
        loss = torch.cat(self.loss_list).sum()

        # loss 들을 다 더한 것을 loss list의 길이로 나누면 loss들의 평균이다. 즉, loss의 평균을 구한다.
        # loss를 self.loss_list의 length로 나눠주는데, 이걸 나눠주지 않아도 학습은 된다. 그런데, 나눠서 loss의 평균을 이용하면 학습이 더 안정적으로 잘 된다.
        # 왜냐하면, 매 episode마다 길이가 다른데, 어떤 episode는 길이가 200이면, loss 200개가 더해져서 gradient가 계산되고,
        # episode 길이가 10 이면, loss 10개가 더해져서 gradient가 계산되니까 episode마다 loss scale이 다르다.
        # 그래서 평균을 취해줌으로써 scale을 다 같게 만들어준다.
        # 결국 학습이 더 안정적으로 된다.
        loss = loss/len(self.loss_list)

        # 모델 매개변수의 gradient를 재설정한다. 
        # 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정합니다.
        self.optimizer.zero_grad()

        # 예측 손실(prediction loss)을 역전파한다. 
        # PyTorch는 각 매개변수에 대한 loss의 gradient를 저장한다.
        loss.backward()

        # 역전파 단계에서 수집된 gradient로 매개변수를 조정한다.
        self.optimizer.step()

        # 다음 episode의 loss_list를 쌓을 수 있게 비워준다.
        self.loss_list = []


def main():
    # 학습 환경
    env = gym.make('CartPole-v1')
    
    # 인스턴스 생성
    model = ActorCritic()
    
    # discounted rate
    gamma = 0.99
    
    # average time step during 20 episodes
    avg_t = 0

    # episode 
    for i_episode in range(10000):
        
        # 상태 초기화
        obs = env.reset()

        # time step
        for t in range(600):
            
            # 상태 값을 numpy에서 tensor형태로 변환 
            obs = torch.from_numpy(obs).float()

            # Actor-Critic이기 때문에, Actor와 Critic이 둘다 있어야한다.
            # model에다가 observation을 input으로 넣으면 output으로는 확률분포(pi)과 value(v) "두 개"가 나온다.
            pi, v = model(obs)

            # pi는 왼쪽 또는 오른쪽의 확률 분포이고, Categorical은 확률 분포를 다루기 위한 함수이다.
            m = Categorical(pi)

            #---------------animation----------------------
            env.render()
            #----------------------------------------------

            # pi 확률 분포를 토대로 action을 구한다.
            action = m.sample()

            # state transition
            obs, r, done, info = env.step(action.item())

            # next state인 obs를 model에다가 input으로 넣어준다. 그러면 next state에서의 value v가 나온다.
            # next state에서 pi는 쓰이지 않아서 언더바 _로 설정해준다.
            _, next_v = model(torch.from_numpy(obs).float())

            # next state에서의 value를 알면 delta를 계산할 수 있다.
            # delta = TD error
            # "delta" = delta^(pi_theta), "next_v" = V^(pi_theta)(s'), "v" = V^(pi_theta)(s)
            delta = r + gamma * next_v - v

            # loss 계산
            # "torch.log" = log, "pi[action]" = pi_theta(s, a), "delta.item()" = delta^(pi_theta)
            # delta.item() 같은 것들은 구현할 때 굉장히 중요한 issue이다. 그냥 delta로 쓰면 학습이 안된다.
            # TD error는 policy pi를 update할 때, 그냥 숫자(상수)이다. 그런데 코드에서의 delta는 네트워크이므로 back propagation을 할 때, 같이 update되므로,
            # delta.item()으로 사용해줌으로써 back propagation을 할 때 update되지 말고, 그냥 log(pi)에다가 상수배만 해주라는 의미로 사용하는 것이다.
            # + 왼쪽 term = policy의 loss 함수이고, + 오른쪽 term = value의 loss 함수이다.
            # policy의 loss 함수에 -가 붙는 이유는 gradient ascent이기 때문이고, value의 loss 함수(= mean squared error)는 gradient descent이므로 -를 붙이지 않는다.
            # 결론: loss 하나에다가 policy의 loss 함수, value의 loss 함수를 다 더함.
            loss = -torch.log(pi[action]) * delta.item() + delta * delta

            # TD actor-critic은 매 step 마다 update할 수 있다. episode가 끝날 때까지 기다리지 않아도 된다.
            # 한 step에서 update가 되면, 그 다음 step에서는 다른 policy로 data(=experience)를 쌓을 수 있다.
            # 그런데 이 코드에서는 이렇게 안하고, model에다가 loss들을 다 모아줬다. 그래서 REINFORCE와 마찬가지로 한 episode가 다 끝난 다음에 update를 했다.
            # 왜냐하면 그냥 코딩적인 issue인데, 그냥 step마다 update를 해도 되는데, 학습이 불안정하다. 그래서 다 모아서 update를 했다.
            # REINFORCE에서의 put_data와 비슷한 역할
            model.gather_loss(loss)

            # done은 엎어지거나 또는 500 step이 지나가면 True가 나온다. 그 전까지는 계속 False이다.
            # 즉, 만약에 끝났으면 "for t in range(600):" 이 for loop을 멈추라는 코드이다.
            if done:
                break
        
        # avg_t의 초기값이 0인데 t가 계속 더해진다. t가 뭐냐면, 몇 step가서 넘어졌나. 몇 step가서 episode가 종료가 되었나를 나타내는 값이다.
        # avg_t는 아래 코드에서 쓰인다.
        avg_t += t

        # 한 episode만큼 데이터가 쌓였으니까, 모델을 학습시키라고 호출하는 코드이다.
        # 그럼 이제 막~~~~ 학습을 시킬 것이다.
        model.train()

        if i_episode % 20 == 0 and i_episode != 0:
            print('# of episode : {}, Average timestep : {}'.format(i_episode, avg_t/20.0))
            avg_t = 0
    
    env.close()

if __name__ == '__main__':
    main()