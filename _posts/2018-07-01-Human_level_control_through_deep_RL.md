---
header: Human-level control through deep reinforcement learning
mathjax: True
---
이 논문은 2013년의 “Playing Atari with Deep Reinforcement Learning” 연구 내용을 좀더 정교하게 다듬고 새로운 내용 몇가지를 추가해 2015년 5월에 Nature 에 게재된 논문입니다. 따라서 본 리뷰에서는 2013년 연구와 중복되는 내용은 대부분 배제하고 새로운 내용 위주로 정리하였습니다. 2013년 연구에 대한 리뷰는 직전의 포스팅을 참고하시면 됩니다. 이 논문에서 새로 추가된 내용의 핵심은 target Q-network, Error clipping 등이며, 2013년보다 더 깊은 네트워크와 학습 시간을 이용하여 더 많은 게임에 DQN 알고리즘을 적용하였습니다.

<!--break-->

***
### 0. Novelty
2013년 연구와 비교할때 대표적인 novelty는 target Q-network 입니다. 

강화학습은 action-value function (Q function) 을 표현하기 위해 신경망과 같은 nonlinear function approximator를 사용할 경우 불안정하거나 발산할 수 있는 것으로 알려져 왔습니다. 그 이유로는 1) 관찰하는 시퀀스 간의 상관관계, 2) Q에 대한 작은 업데이트가 policy를 크게 변화시키고 data distribution을 변화시켜 버림, 3) action-value (Q) 와 target value 간의 상관관계 등이 있습니다. 

이에 대한 해법은 두가지입니다. 첫번째는 experience replay로, 이에 대한 자세한 설명은 2013년 논문 리뷰를 참고하기 바랍니다. 두번째 해법은 별도의 target Q-network 를 준비함으로써 action-value (Q) 와 target value 간의 상관관계의 문제를 해결하는 것입니다. 별도의 target Q-network는 다음과 같이 진행합니다. 

C번째 업데이트 시마다 network Q를 복제하여 target network Q 을 만듭니다. Q 을 이용하여 그 다음의 C번의 업데이트를 위한 Q-learning target인 yi 를 생성합니다. 이런 과정이 없다면 Q(st, at) 를 증가하도록 업데이트하면 Q(st+1, at) 역시 증가하기 때문에 policy의 oscillation 또는 divergence를 유발하게 되기 때문입니다. 

다시 말해 Q value가 Q value 자신에 의존하여 업데이트를 하는 식의 재귀적 (recursive) 방식이기 때문입니다. 쉽게 말해서 target에 접근하도록 업데이트를 해줬는데 정작 target은 다시 도망가는 모양새인 것입니다. 

기존의 파라미터를 이용하여 타겟을 생성함으로써 Q에 대한 업데이트가 이루어지는 시간과 이 업데이트가 타겟 yj에 영향을 미치는 시간 사이에 지연을 줌으로써 oscillation이나 divergence가 생길 가능성을 낮춰주는 것입니다. 


### 1. Methods 
#### (1) Preprocessing

아타리 에뮬레이터는 그 성능의 한계로 인해 일부 오브젝트들은 홀수 프레임에만, 다른 오브젝트들은 짝수 프레임에만 존재하게 설계가 되어 있습니다. 따라서 본 연구에서는 현재의 프레임과 직전의 프레임에 걸쳐 각 픽셀 컬러 값의 최대 값을 취하는 방식으로 하나의 프레임을 인코딩하였습니다. 색상을 제거하여 gray color로 변환하고 84 x 84 의 크기로 down-sampling 해 주었습니다.

Q-function에 입력은 m개의 최근 프레임을 스택하여 이용하였습니다. m의 값으로는 4를 이용하였으나 3이나 5를 써도 결과는 동일하게 잘 나온다고 합니다. 

#### (2) Source code 

소스코드는 https://sites.google.com/a/deepmind.com/dqn/ 에 있으며 비상업적 용도로만 사용 가능합니다. 

#### (3) Model architecture

입력은 state만 주고, 이 state에서 가능한 모든 action에 대한 Q-value를 출력하게 함으로써 single forward pass만으로 Q-value를 연산하도록 하였습니다.

Model architecture는 2013년 버전에서는 2개의 convolutional layer를 사용한데 반해, 2015년 버전에는 3개를 사용하였으며, 각 conv layer의 필터수가 2배로 증가하였고 fully connected layer의 레이어의 수도 2배로 증가하였습니다. 다시 말해 model architecture가 2배 이상으로 커졌습니다. 

2013년 버전 아키텍처

* Input layer: 함수 Φ에 의해 생성된 84 x 84 x 4 의 이미지
* Conv layer #1: 16 filters of 8 x 8 with stride 4 + ReLU
* Conv layer #2: 32 filters of 4 x 4 with stride 2 + ReLU
* Fully connected layer: 256 + ReLU
* Output layer: Fully connected layer with a single output for each valid action (4개 ~ 18개)

2015년 버전 아키텍처

* Input layer: 함수 Φ에 의해 생성된 84 x 84 x 4의 이미지
* Conv layer #1: 32 filters of 8 x 8 with stride 4 + ReLU
* Conv layer #2: 64 filters of 4 x 4 with stride 2 + ReLU
* Conv layer #3: 64 filters of 3 x 3 with stride 1 + ReLU
* Fully Connected layer: 512 + ReLU
* Output layer: Fully connected layer with a single output for each valid action (4개 ~ 18개)

<figure>
	<img src="/img/3/architecture.png" alt="alt text">
</figure>	

#### (4) Training details 

(DQN의 성능 vs. 기존 알고리즘의 성능 vs. 사람의 성능) DQN 의 성능을 기존 알고리즘의 성능, 사람과의 성능과 비교한 결과가 아래 그림에 나와있습니다. 2013년에는 아타리 2600 게임 7개에 대해 성능을 테스트하였으나, 이번 논문에서는 총 49개의 게임에 대해 비교하였습니다. 

그림에서 각 게임에 나와있는 수치는 (DQN의 점수 - 랜덤플레이 점수)/ (프로게이머의 점수 - 랜덤플레이 점수) 에 100을 곱해 계산한 것입니다. 다시 말해 0%는 DQN의 점수 (파란색 부분) 가 랜덤플레이의 수준이라는 의미이며, 100%는 프로게이머 수준이라는 것입니다. 일반적인 게이머의 수준은 프로게이머의 75% 수준으로 가정하였습니다. 회색 부분은 기존의 선형 함수 기반 학습 알고리즘을 이용하여 테스트했을 때의 점수를 의미합니다. 

<figure>
	<img src="/img/3/atari_result.png" alt="alt text">
</figure>	

모든 게임에 대해 동일한 네트워크 아키텍처, 학습 알고리즘, 하이퍼파라미터를 사용하였습니다. 하이퍼파라미터의 설정은 아래 표를 참고하도록 합니다. 

<figure>
	<img src="/img/3/hyperparameters.png" alt="alt text">
</figure>	

표에서 보다시피 대부분의 하이퍼파라미터의 설정은 2013년 연구의 설정과 대동소이합니다. 특이한 파라미터로는 1) target Q-network의 업데이트는 10,000번마다 진행한다는 점, 2) replay는 50,000개의 replay가 메모리에 쌓인 다음부터 활용하기 시작한다는 점 (즉, 50,000개가 쌓이기까지는 랜덤 플레이), 3) 에피소드 시작 시 에이전트는 최대 30번까지 no-op (아무런 동작도 하지 않음) 하게 한다는 것 정도입니다. 

게임에 준 유일한 변형은 학습 시의 보상 체계입니다. 모든 양의 보상은 +1, 음의 보상은 -1로 설정함으로써 reward clipping을 적용한 것입니다. Reward clipping 관련된 자세한 내용은 직전에 포스팅한 2013년 연구에 대한 리뷰를 참고하기 바랍니다. 

신경망의 optimizer로는 RMSProp 알고리즘을 사용하였으며, 미니배치의 수는 32를 사용하였습니다. 학습중의 behavior policy로는 ε-greedy 방식을 사용하였는데, 역시 자세한 내용은 2013년 연구 리뷰를 참고하면 됩니다. 

학습은 2013년에는 1,000만 프레임에 대해 진행했는데, 이번에는 5,000만 프레임에 대해 진행하였습니다. 5,000만 프레임은 게임 시간으로 따지면 38일동안 꼬박 게임을 한 것과 동일한 분량입니다. Replay memory에는 가장 최근 프레임이 100만개 포함될 수 있도록 하였습니다. 

Frame skipping 기술도 활용을 하였는데, 역시 상세 내용은 2013년 연구 리뷰를 참고하시기 바랍니다.

하이퍼파라미터 값과 optimization 파라미터 값은 Pong, Breakout, Seaquest, Space Invaders, Beam Rider 등 5개의 게임에 대한 informal search를 통해 정했습니다. 정식으로 하이퍼파라미터 값을 정하기 위해서는 systematic grid search를 수행해야 했지만, 연산 비용이 너무 높기 때문에 수행하지 못 했습니다. 

실험은 아래와 같은 게임에 대한 최소한의 지식만을 이용하여 진행하였습니다.

* 화면 이미지
* 점수 (게임마다 다르나 변형 없이 사용)
* 액션의 수 
* 생명의 수 (예를 들어 Breakout의 경우에는 5개)

#### (5) Evaluation procedure

학습을 마친 에이전트는 각각의 게임을 30회씩 (최대 시간 5분) 플레이하여 테스트를 진행하였습니다. 테스트 시에 오버피팅이 발생하는 가능성을 최소화하기 위해 ε-greedy policy를 적용하였으며, 이때 ε의 값은 0.05를 사용하였습니다. 

Baseline comparison으로는 랜덤 에이전트를 사용하였는데, 사람이 어떤 버튼을 누를 수 있는 가장 빠른 속도인 10Hz (6번째 프레임) 마다 랜덤한 액션을 선택하도록 하고 이후 다음 선택 지점 (6 프레임 후) 까지는 동일한 액션을 반복하도록 하였습니다. 사실 60Hz (매 프레임) 마다 액션을 선택하도록 랜덤 에이전트를 실험하기도 하였으나 겨우 6개의 게임에서만 성능이 5% 정도 향상되었을 뿐 특별한 효과는 없었습니다. 

사람 테스터는 중간에 게임을 멈추거나 저장하거나 리로드할 수 없게 하였으며 게임 사운드는 끈 상태에서 진행함으로써 에이전트의 환경과 유사하게 맞추어 주었습니다. 사람의 성능은 각각의 게임에 대해 2시간동안 연습하게 한 다음에 실제 테스트는 20번 (최대 시간 5분) 진행하여 측정하였습니다. 

#### (6) Algorithm

알고리즘은 2013년 논문 리뷰의 2. Background 부분과 동일하니, 이를 참고하기 바랍니다. 

#### (7) Training algorithm for DQN

전체 알고리즘은 아래와 같습니다. 

<figure>
	<img src="/img/3/algorithm.png" alt="alt text">
</figure>	

2013년 연구 당시의 알고리즘과 비교하면 딱 한가지 차이만 존재합니다. 바로 target Q-network의 설정과 관련된 3행과 15행 입니다.  

* 3행에서는 target action-value function Q hat 의 가중치 θ-를 θ로 초기화하고 있으며, 
* 15행에서는 C step마다 Q 을 Q로 리셋해 줍니다.

기존의 Q-learning 알고리즘과의 차이점은 크게 아래 세가지입니다. 

#### (1) Experience replay의 사용

Experience replay 관련된 알고리즘에 대해서는 2013년 논문에서 했던 언급을 그대로 다시 언급하고 있기 때문에 2013년 연구 리뷰를 참고하기 바랍니다. 

#### (2) Target Q-network

Target Q-network 관련해서는 앞의 0. Novelty 부분의 설명을 참고하도록 합니다. 

#### (3) Error Clipping

Error (loss) 를 -1과 1 사이에 오도록 하는 것이 알고리즘의 안정성에 도움이 된다는 사실을 발견하였습니다. Error clipping은 Huber loss function을 이용하여 구현 가능합니다. 


### 2. 성능 비교 (Experience replay, Target Q-Network)

아래 표에서 보다시피 experience replay와 target Q-network를 가미함으로써 성능이 비약적으로 향상된 것을 확인할 수 있습니다.

<figure>
	<img src="/img/3/performance.png" alt="alt text">
</figure>	

### 3. 성능 비교 (DQN vs. Linear function approximator)

대부분의 게임에서 DQN은 linear function approximator보다 월등한 성능을 보였습니다. 

<figure>
	<img src="/img/3/performance2.png" alt="alt text">
</figure>	

### 4. Value function 시각화

#### (1) Breakout에서의 value function 추이

게임 화면의 블록을 자세히 보게되면, 아래쪽 블록을 깨는 ①과 ②에서는 17 정도의 state value를 보이며, 위쪽 블록을 깨는 ③에서는 state value가 21 정도로 올라가는데, 그 이유는 이제 터널을 뚫고 여러개의 블록을 한꺼번에 깰 수 있다는 기대감이 반영되었기 때문입니다. 마지막으로, 터널을 뚫고 들어간 ④에서는 state value가 23 이상으로 올라가는 것을 확인할 수 있습니다. 

<figure>
	<img src="/img/3/value_fn_breakout.png" alt="alt text">
</figure>	

#### (2) Pong 에서의 action-value function (Q-value function) 추이

아래 그림을 보면, ①에서는 NO-OP, UP, DOWN에 대한 Q-value가 모두 0.7 수준입니다. ②에 가면 위로 이동해서 공을 맞춰야 하기 때문에 UP의 Q-value가 높고, NO-OP과 DOWN의 Q-value는 매우 낮은 것을 확인할 수 있습니다. ③에서는 조금 더 위로 올라가야 하기 때문에 여전히 UP의 Q-value가 높게 나타납니다. ④에서는 공이 상대방의 paddle을 넘어서 보상 +1을 따기 직전이기 때문에 NO-OP, UP, DOWN 모두에 대해 Q-value가 1로 나타납니다. 

<figure>
	<img src="/img/3/value_fn_pong.png" alt="alt text">
</figure>	

### 5. t-SNE 시각화 1

아래 그림은 t-SNE 알고리즘을 실행하여 DQN의 마지막 hidden layer를 표현한 것입니다. 각 점의 색은 각 state에 대한 state value (V, state의 최대 기대 보상) 를 나타내며 진홍색은 높은 state value를, 군청색은 낮은 state value를 의미합니다. 

게임 화면이 가득 차 있는 경우 (우측 상단) 와 거의 비어 있는 경우 (좌측 하단) 에는 state value가 높은데, 그 이유는 우리의 DQN이 화면을 비우면 (즉, 게임을 클리어하면) 화면이 가득 찬 화면으로 이어진다 (즉, 새로운 다음 게임이 시작된다) 는 점을 학습하였기 때문입니다. 이에 비해 중간쯤 진행된 경우 (중앙 하단) 에는 즉각적인 보상이 작기 때문에 state value가 낮습니다. 

우측 하단, 좌측 상단, 중앙 상단의 게임화면을 보면 앞의 예들과는 다르게 화면의 구성에 다소 차이가 있음에도 불구하고 유사한 state value를 가집니다. 그 이유는 게임 화면을 자세히 보면 오렌지색의 벙커의 존재 여부에 의해 달라보이는 것인데, 이 벙커는 게임을 클리어하는데 그다지 중요하지 않기 때문입니다. 

<figure>
	<img src="/img/3/t-sne1.png" alt="alt text">
</figure>	

### 6. t-SNE 시각화 2

이 그림은 사람의 플레이와 DQN의 플레이의 조합에 대해 t-SNE 알고리즘을 실행하여 DQN의 마지막 hidden layer를 표현한 것입니다. 사람의 플레이는 오렌지색, DQN의 플레이는 파란색으로 표현하였는데, 유사한 state 를 가지는 화면은 사람의 플레이와 DQN의 플레이에서 유사하게 나타난다는 점에서 DQN에 의해 학습된 표현이 실제로 일반화하고 있다는 것을 보여주고 있습니다. DQN 에이전트가 사람의 플레이에서 발견되는 것과 유사한 state의 시퀀스를 따른다는 것입니다. 

<figure>
	<img src="/img/3/t-sne2.png" alt="alt text">
</figure>	








### 2. Background
 
강화학습을 위한 요소는 아래와 같이 정리할 수 있습니다. 

* Agent
* Environment $\mathcal{E}$ (Atari emulator)
	* Stochastic
	* Partially Observable
		* 에이전트는 현재의 스크린만 볼 수 있습니다. 
		* 에이전트는 에뮬레이터의 내부 상태를 관찰할 수 없습니다. 
* Actions $\mathcal{A} = \\{1, . . . , K \\}$
* Observations $\mathcal{X}$
* Rewards $\mathcal{R}$
	* 보상은 게임에서 얻는 점수인데, 지연된 보상의 형태를 가집니다. 

몇가지 정의

* 환경은 stochastic 하기 때문에 어떤 상태에서 특정 액션을 취할때마다 보상이 달라질 수 있습니다. 
* 에이전트는 현재의 화면만 볼 수 있기 때문에 partially observable한 환경입니다.
* 한개의 화면만 보고서는 상황 파악이 불가능하기 때문에 몇개의 화면을 묶어서 시퀀스 기반으로 학습을 진행합니다. <br> ($s_t = x_1,a_1,x_2,..., a_{t-1}, x_t$). 즉, 하나의 상태는 하나의 시퀀스가 됩니다.
* 게임은 언젠가는 끝납니다. 결국 유한한 MDP 문제를 풀어야 합니다. 

현재의 화면만 보고서는 상황 파악이 불가능한 이유는 쉽게 말하면 게임에서 하나의 화면만 봐서는 이 공이 위로 가는지 아래로 가는지 알 수가 없기 때문입니다. 
강화학습은 결국 MDP 문제를 푸는 방식입니다. MDP는 Markov property를 따라야 하는데, Markov property (또는 Markov state) 란 “The future is independent of the past given the present” 인 상태, 다시 말해서 현재의 state만 보고 미래를 예측할 수 있어야 하는 것이다. 따라서 한 화면만을 state로 이용하면 Markov state를 깨게 되는 것입니다. 따라서 본 연구에서는 이에 대한 대안으로 4개의 프레임을 하나의 state로 설정하였습니다. (2개의 프레임을 사용하면 속도가 파악 가능하고, 3개의 프레임을 사용하면 가속도까지도 파악 가능할 것입니다.)

이제 우리는 이렇게 정의한 st를 이용하여 MDP에 대한 표준적인 강화학습 방법을 적용할 수 있습니다. 

에이전트의 목적은 미래의 보상을 최대화하도록 하는 액션을 선택하는 것입니다. 미래의 할인된 보상은 다음과 같이 정의합니다.
<center> $R_t = \displaystyle \sum_{t'=t}^T \gamma^{t'-t} r_t'$ </center> <br>

Optimal action-value function (= Optimal Q-function) 은 아래와 같이 정의합니다. 
<center> $Q^{\ast}(s,a) = \max_{\pi} \mathbf{E}[R_t|s_t=s, a_t=a, \pi]$ </center> <br>

이 optimal action-value function은 아래와 같이 정의되는 Bellman equation을 따릅니다.

<center> $Q^{\ast}(s,a) = \mathbf{E}_{s' \sim \mathcal{E}} \big [r+\gamma \max_{a'} Q^{\ast}(s',a')|s,a \big]$ </center> <br>

강화학습의 기본적인 아이디어는 value iteration 알고리즘으로, Bellman equation을 반복적으로 업데이트로 이용함으로써 action-value function을 예측하는 것입니다. 

<center> $Q_{i+1}(s,a) = \mathbf{E} \big [r + \gamma \max_{a'} Q_i(s',a')|s,a \big ]$ </center> <br>

Value iteration 알고리즘에 의하면 업데이트를 무한번 시도하면 optimal action-value function으로 수렴하게 됩니다.

<center> $Q_i \rightarrow Q^{\ast} \text{as} \ i \rightarrow \infty$ </center> <br>

하지만 사실 이러한 기본적인 접근방법은 실용적이지 않습니다. 왜냐하면 action-value function은 각각의 시퀀스에 대해 따로 예측되고 일반화되지 못하기 때문입니다. 따라서 action-value function을 예측하기 위해서는 function approximator를 이용하는 것이 일반적입니다. 

<center> $Q(s, a; \theta) \approx Q^{\ast}(s,a)$ </center> <br>

강화학습에서는 일반적으로 function approximator로 linear function approximator가 사용되었지만 때로는 신경망과 같은 non-linear function approximator가 사용됩니다. 가중치 θ를 가지는 신경망 기반 function approximator를 Q-Network 라고 합니다. Q-network 는 반복 시마다 변화하는 일련의 loss function을 최소화함으로써 학습시킬 수 있습니다.

<center> $L_i(\theta_i) = \mathbf{E}_{s,a \sim p(\cdot)} \big [ (y_i-Q(s,a;\theta_i))^2 \big]$ </center> <br>

여기서 yi는 아래와 같이 i번째 반복 시의 타겟입니다.

<center> $y_i = \mathbf{E}_{s' \sim \mathcal{E}} \big[ r+ \gamma \max_{a'} Q(s', a'; \theta_{i-1})|s, a \big]$ </center> <br>


$ρ(s, a)$ 는 behavior distribution이라고 부르는 것으로, 시퀀스 $s$와 액션 $a$에 대한 확률 분포입니다. 

지도학습에서 사용되는 타겟은 학습이 시작되기 전에 고정되지만, 강화학습의 타겟은 네트워크의 가중치의 영향을 받습니다. Loss function을 가중치에 대해 미분하면 다음과 같은 gradient를 얻게 됩니다.

<center> $\nabla_{\theta_i} L_i(\theta_i) = \mathbf{E}_{s,a \sim p(\cdot); s' \sim \mathcal{E}} \bigg [ \big ( r + \gamma \max_{a'} Q(s', a';\theta_{i-1}) - Q(s,a;\theta_i) \big) \nabla_{\theta_i} Q(s,a;\theta_i) \bigg ]$ </center> <br>


이 gradient에서처럼 full expectation을 계산하는 것보다는 stochastic gradient descent에 의해 loss function을 최적화하는 것이 신속한 연산을 가능하게 합니다. 만약 가중치가 매번 업데이트된다면 기대치는 behavior distribution과 환경 E 각각으로부터 하나의 샘플에 의해 대체될 것이며, 이 과정을 거쳐 우리의 Q-learning 알고리즘에 도달하게 됩니다.

이 알고리즘의 특성은 다음과 같습니다.
 
* Model-free
	* 명시적으로 환경의 예측을 구축하는 것이 아니라, 환경으로부터의 샘플을 이용하여 직접적으로 강화학습 문제를 해결하는 방법을 취하기 때문에 model-free 입니다. 
	* model-based 는 기존에 학습시켜 높은 모델을 이용하는데 반해, model-free 는 직접 시행착오를 거쳐 진행합니다.
* Off-policy
	* 적절한 exploration을 보장하는 behavior distribution 을 따르면서 아래와 같은 greedy strategy 에 대해 학습합니다.
	* On-policy에서는 에어전트의 action (exploration을 포함) 에 의해 최적의 policy를 학습하는데 반해, off-policy 에서는 에이전트의 action과는 독립적으로 behavior distribution 을 따르면서 최적의 policy를 학습합니다.

<center>$a = \max_a Q(s, a;\theta)$</center> <br>
 
### 3. Related Work
TD-gammon 게임에 Q-learning과 유사한 model-free 강화학습 알고리즘을 적용하고 하나의 히든 레이어를 가지는 다계층 퍼셉트론을 사용하여 value function 을 approximate함으로써 사람을 이기는 수준의 성과는 이미 오래 전인 1995년 경에 달성한 바 있습니다. 하지만 동일한 방법을 체스, 바둑, 체커 등에 적용하였으나 실패하였으며 이로 인해 사람들은 강화학습이 성공했던 것은 TD-gammon에 국한된 특별한 케이스였다는 생각을 하게 되었습니다. TD-gammon에 유독 잘 통했던 이유는 다음과 같이 생각되었습니다.
 
* 주사위를 굴릴 때의 랜덤성 (stochasticity) 이 다양한 상태를 탐험 (explore) 하도록 도움
* 주사위를 굴릴 때의 랜덤성 (stochasticity) 이 value fn을 부드럽게 만들어 줌

또한 Q-learning과 같은 model-free 강화학습 알고리즘에 nonlinear function approximator 또는 off-policy 학습을 사용하면 Q-network가 발산해 버렸습니다. 그래서 안전하게 linear approximator를 사용해서 조금이라도 수렴이 잘 되도록 개선하는 데에 초점을 맞추어 왔습니다. 

하지만 최근들어 딥러닝에 대한 관심이 높아지면서 개선된 시도가 시작되었습니다. 발산 이슈는 gradient temporal-difference 방법을 이용하여 부분적으로는 해결되기는 하였지만 여전히 비선형 문제를 근본적으로 해결하지 못 하고 있었습니다. 

그나마 가장 근접한 시도는 NFQ (Neural Fitted Q-learning) 입니다. NFQ는 RPROP 알고리즘으로 Q-network의 파라미터들을 업데이트하는데, 데이터셋의 크기에 비례하여 연산 비용이 드는 배치 업데이트 방법을 사용했습니다. 반면 우리는 stochastic gradient update를 이용함으로써 큰 데이터셋에도 대응할 수 있는 낮은 연산 비용을 가져갈 수 있습니다. 또한 NFQ는 비디오의 입력만을 이용하여 오토인코더를 통해 저차원의 표현을 성공적으로 학습하게 하고나서 NFQ를 이 표현에 적용하였으나, 우리는 이와는 대조적으로 강화학습을 직접 비디오 입력부터 적용하는 end-to-end 방식이기 때문에 action-values와 직접 연관된 피쳐를 학습할 수 있습니다. Q-learning 역시 이미 experience replay와 단순한 신경망과 조합된 바 있으나 저차원의 상태로부터 시작하으며, 우리처럼 raw visual input 으로부터 시작한 것은 아니였습니다. 

또한, 아타리 2600 에뮬레이터를 강화학습의 플랫폼으로 사용한 사례들은 이전에도 있어왔습니다. 

### 4. Deep Reinforcement Learning
본 연구의 목표는 강화학습 알고리즘을 DNN에 연결하는 것입니다. 

TD-Gammon 아키텍처에서 시작합니다. 여기에서는 experience의 on-policy 샘플들 ($s_t$, $a_t$, $r_t$, $s_{t+1}$, $a_{t+1}$) 로부터 value function을 예측하는 네트워크의 파라미터들을 업데이트합니다. 이러한 접근방법으로 이미 20년전에 TD-Gammon에서 사람을 이긴 바 있기에, 20년간의 하드웨어의 발전과 DNN 등을 고려하면 더 좋은 결과를 낼 수 있을 것으로 예상됩니다.

TD-Gammon과는 달리 우리는 experience replay 기법을 사용해서 에이전트의 각 단계에서의 경험 $e_t = (s_t, a_t, r_t, s_{t+1})$ 을 replay memory (데이터셋 $D = e_1, ..., e_N)$ 에 저장합니다. 알고리즘의 내부 반복문이 돌아갈때 Q-learning 업데이트 (미니배치 업데이트) 를 경험 샘플 $e \sim D$ 에 적용합니다. 경험 샘플은 저장된 샘플들 중에 랜덤하게 추출합니다. Experience replay를 수행한 후에 에이전트는 $\epsilon$-greedy 정책에 의해 액션을 선택하고 실행합니다. 신경망에 입력으로 임의의 길이의 history를 넣어주는 것은 어려움이 있을 수 있기 때문에 우리의 Q-function 은 함수 $\Phi$ 에 의해 생성된 정해진 길이의 history를 이용합니다. 우리가 Deep Q-Learning이라고 명명한 전체 알고리즘은 아래와 같습니다. 

<figure>
	<img src="/img/2/Algorithm.png" alt="alt text">
</figure>	

이러한 접근방법은 표준 Q-learning에 비해 몇가지 장점을 가집니다.
각 단계에서의 experience가 가중치 업데이트에 여러번 사용됨으로써 데이터의 효율성을 기할 수 있습니다.
연속된 샘플들에서 직접 학습을 하는 것은 이 샘플간의 강한 상관관계 때문에 비효율적입니다. 샘플링을 랜덤하게 함으로써 이러한 상관관계를 깨고 업데이트의 variance를 줄일 수 있습니다.
On-policy 학습을 할때 현재의 파라미터들이 다음 데이터 샘플을 결정하는데, 이 다음 데이터 샘플들은 그 파라미터들이 학습된 샘플들입니다. 예를 들어 만약 최대화 액션이 왼쪽으로 움직이는 것이면 학습 샘플은 왼쪽의 샘플에 의해 좌지우지되게 될 것입니다. 만약 최대화 액션이 오른쪽으로 움직이면 학습 분포 역시 이동하게 됩니다. 원치 않는 피드백 루프가 생기고 파라미터들은 local minimum에 빠지거나 재앙적으로 발산할 수 있습니다.
Experience replay를 사용함으로써 기존의 상태들에 의해 behavior distribution이 평균화되고 학습을 부드럽게 하며 파라미터들이 심하게 흔들리거나 발산하는 것을 피할 수 있습니다. Experience replay로 학습 시에는 (현재의 파라미터들은 샘플을 생성하는데 사용되는 파라미터들과 다르기 때문에) Q-learning의 선택에 모티베이션을 주는 off-policy를 학습할 필요가 있다는 사실을 잊지 말아야 합니다. 

Replay memory에는 가장 최근의 $N$개의 experience tuple만 저장한 후, 업데이트 수행시에 $D$로부터의 샘플링은 랜덤하게 균일하게 합니다. 이 접근법은 사실 다소 제한된 방식입니다. 왜냐하면 1) 메모리 버퍼가 중요한 transition을 구별하지 못하며, 2) 유한한 메모리 크기 ($N$) 로 인해 항상 최신의 transition으로 덮어쓰기 때문입니다. 균일한 샘플링은 replay memory 내의 모든 transition에 동일한 중요도를 부여합니다. 보다 정교한 샘플링 전략은 마치 prioritized sweeping에서와 같이 우리가 가장 많이 학습을 한 transition에게 더 큰 중요도를 부여하는 것이 될 것입니다. 

#### 4.1 Preprocessing and Model Architecture

##### (1) 전처리

* 아타리 프레임 원본
	* 210 x 160 pixel images with a 128 color palette
* 전처리
	* Gray-scaling
	* Down-sampling to 110 x 84 
	* Cropping for Input representations : 84 x 84 (AlexNet이 정사각형을 원하기 때문)

마지막 4개의 프레임에 이러한 전처리를 함수 Φ에 의해 수행한 후 4개의 프레임을 stack하여 Q-function 의 입력으로 생성해 줍니다.

NN을 이용하여 $Q$를 파라미터화하는 방법에는 몇가지 방법이 있습니다. $Q$는 history-action 쌍을 $Q$-value (스칼라) 예측값과 맵핑하기 때문에 기존의 접근방법에서는 입력으로 history와 action을 사용했습니다. 이 방법의 단점은 각 액션에 대한 $Q$-value를 계산하기 위해 별도의 forward pass 가 필요하기 때문에, 액션의 수가 증가함에 따라 연산비용도 비례하여 증가한다는 것입니다. 우리는 이대신에 state만 입력으로 넣어주고 각 액션에 대해 별도의 출력 유닛이 있는 아키텍처를 사용하였습니다. 출력은 입력되는 state에 대한 개별 액션 별로 예측되는 $Q$-value가 오게됩니다. 이 방법의 장점은 단 한번의 forward pass로 주어진 상태에서 가능한 모든 액션에 대한 $Q$-value를 계산할 수 있다는 것입니다. 

##### (2) 아키텍처
* Input layer: 함수 Φ에 의해 생성된 84 x 84 x 4 의 이미지
* Conv layer #1: 16 filters of 8 x 8 with stride 4 + ReLU
* Conv layer #2: 32 filters of 4 x 4 with stride 2 + ReLU
* Fully connected layer: 256 + ReLU
* Output layer: Fully connected layer with a single output for each valid action (4개 ~ 18개)

참고로 아타리 조이스틱에서 취할 수 있는 최대 액션 수는 18가지라고 합니다. 우리가 주로 보고 있는 Breakout 게임에서는 NOOP, LEFT, RIGHT, FIRE 등 총 4가지 액션이 존재합니다.  

우리는 이와 같이 우리의 방식에 따라 학습된 CNN을 DQN (Deep Q-Network) 이라고 명명합니다.

### 5. Experiments
##### Reward Clipping
게임에 적용한 단 한가지 변화는 보상 체계입니다. 게임마다 점수의 스케일이 다르기 때문에 양의 보상은 1, 음의 보상은 -1로 통일시켰습니다. 즉 reward clipping을 한 것인데, 그 이유는 오류의 도함수 (error derivatives) 의 스케일을 제한하고, 모든 게임에 동일한 learning rate를 적용할 수 있기 때문입니다. 물론 보상의 강도를 제한함으로써 에이전트의 성능에 제약을 가져올 수는 있습니다. 

##### Hyperparameters (모든 게임에 공통)

최적화 알고리즘으로는 RMSProp을, Minibatch의 크기는 32를, 학습 도중 behavior policy 는 ε-greedy 를 사용하였습니다.
ε 의 값은 처음부터 100만 프레임까지는 1에서 0.1까지 동일한 비율로 감소하게 하였고, 100만 프레임 이후에는 0.1로 고정하였습니다. 초기에는 exploration을 주로 하고, 진행에 따라 exploitation 비중을 늘려준 것입니다. 
학습은 1,000만 프레임까지 진행하였는데, Replay Memory의 사이즈는 100만개이며, 100만개가 가득 차면 가장 오래된 경험부터 메모리에서 사라지게 됩니다. 

##### Frame Skipping : 유일하게 다르게 설정한 Hyperparameter

Frame skipping을 적용함으로써 에이전트가 모든 프레임을 보게할 것이 아니라 $k$번째 프레임을 보고 액션을 취하게 하는데, 스킵한 프레임에서도 이 액션을 반복하게 됩니다. 
액션의 선택을 $k$번 반복하는 것보다 한번만 하는 것이 연산을 훨씬 더 줄여주게 됩니다. 이로 인해 거의 게임 플레이의 속도를 $k$배로 늘려줄 수 있습니다. 
모든 게임에 $k = 4$로 설정하였으나, Space Invader에만 $k = 3$ 으로 설정하였는데, 그 이유는 깜빡깜빡하게 설정되어 있는 레이저가 프레임을 스킵하는 순간에 보이지 않는 경우가 많이 생겼기 때문입니다. 바로 이것이 유일하게 다르게 설정한 hyperparameter입니다. 

OpenAI Gym에서 아타리에 대한 강화학습 모델을 구현 시 ```env = gym.make(‘BreakoutDeterministic-v4’)```를 선택하면 됩니다. ```v4```가 바로 $k=4$ 인 frame skipping을 미리 구현해 놓은 것입니다. `v0`은 랜덤한 frame skipping (2, 3, 4) 이라고 합니다.

#### 5.1 Training and Stability

지도학습에서는 학습 도중에 학습 데이터셋과 검증 데이터셋을 평가함으로써 쉽게 모델의 성능을 추적할 수 있지만, 강화학습에서는 이게 어렵습니다. 왜냐하면 평가 지표가 한번의 에피소드로부터 에이전트가 얻은 총 보상이기 때문입니다. 정책의 가중치에 대한 작은 변화가 해당 정책이 방문하는 상태의 분포에 큰 변화를 야기할 수 있기 때문에 평균 총 보상은 매우 noisy합니다 (아래의 왼쪽 2개 그림). 

보다 안정적인 지표는 정책의 예측 action-value function인 Q-function 입니다. Q-function 은 에이전트가 주어진 상태에서 정책을 따름으로써 얼마만큼의 할인된 보상을 획득할 수 있는지에 대한 예측치를 제공합니다 (아래의 오른쪽 2개 그림). 

우리의 실험에서는 $Q$ 값이 발산하는 경우는 없었습니다.

<figure>
	<img src="/img/2/stability.png" alt="alt text">
</figure>

#### 5.2 Visualizing the Value Function

시각화를 통해 우리의 value function이 꽤 복잡한 일련의 이벤트에 대해 어떻게 진화해 나가야 할지 학습할 수 있음을 보입니다.

<figure>
	<img src="/img/2/seaquest.png" alt="alt text">
</figure>
	
노란색 잠수함이 에이전트이며, 녹색 잠수함들은 적 (환경) 입니다. 

* (A) 적이 왼쪽에 나타나자 예측 $Q$ 값이 상승합니다.
* (B) 에이전트가 어뢰를 발사하고 적을 맞추기 직전 상태에서 예측 $Q$ 값이 정점에 이릅니다.
* (C) 적을 맞추자 예측 $Q$ 값이 원래의 값 수준으로 급락합니다.

#### 5.3 Main Evaluation

아래의 대상들과 비교 평가를 수행하였습니다.

* Sarsa는 기존의 강화학습 방법으로, 아타리에 대해 수동적으로 추출한 피쳐셋에 대해 선형 정책을 학습합니다.
* Contingency 역시 기존의 강화학습 방법으로, Sarsa와 유사하나 피쳐셋을 augmentation 합니다.
* Expert human에 대해서는 2시간동안 연습 게임을 해 본 다음에 측정하였습니다. 
* Random 방식은 말그대로 에이전트가 랜덤한 액션을  선택하게 하여 테스트한 것입니다. 
* HNeat Best 와 HNeat Pixel 은 policy search 접근 방식입니다. 

평가 결과는 아래와 같습니다. 

<figure>
	<img src="/img/2/result.png" alt="alt text">
</figure>

4행의 DQN은 평균 점수, 8행의 DQN Best는 최고 점수를 의미합니다. 게임별 성능을 사람과 비교하면 다음과 같습니다.

* 사람보다 나은 결과가 나온 게임: Breakout, Enduro, Pong
* 사람과 유사한 결과가 나온 게임: Beam Rider
* 사람보다 못한 결과가 나온 게임: Q*bert, Seaquest, Space Invaders

사람보다 못한 결과가 나온 게임들은 네트워크가 긴 시간에 걸치는 전략을 찾아야 하는데 실패했습니다. 

### 6. Conclusion

* 입력으로 raw pixel만을 이용하고 CNN을 이용하여 강화학습시킨 최초의 성공적인 연구입니다.
* DQN 에 experience replay memory 와 stochastic minibatch updates 를 성공적으로 시도하였습니다. 
* 7개의 게임에 대해 동일한 아키텍쳐와 하이퍼파라미터를 이용해 6개의 게임에 대해 최고 수준의 성능을 보였습니다.


### References
will be updated soon...
