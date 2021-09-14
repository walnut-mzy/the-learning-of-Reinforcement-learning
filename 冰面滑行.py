import gym
import numpy as np
env=gym.make("FrozenLake-v0")
env=env.unwrapped
def play_policy(env,policy,render=False):
    total_reward=0.
    observation=env.reset()
    while True:
        if render:
            env.render()
        #print(policy[observation],env.action_space.n)
        action=np.random.choice(env.action_space.n,p=policy[observation])
        #print(action)
        observation,reward,done,_=env.step(action)
        #print(env.step(action))
        total_reward+=reward
        if done:
            break
    return total_reward
env.render()
print("状态空间为：",env.observation_space)
print("动作空间为：",env.action_space)
#
# #求解随机策略
#env.unwrapped.nA动作空间数量
#env.unwrapped.nS状态空间数量
# print((env.unwrapped.nS,env.unwrapped.nA))
random_policy=np.ones((env.unwrapped.nS,env.unwrapped.nA))/env.unwrapped.nA
# print(random_policy)
# eplisode_rewards=[play_policy(env,random_policy) for _ in range(100)]
# print("随机策略 平均奖励：{}".format(np.mean(eplisode_rewards)))
def v2q(env,v,s=None,gamma=1.):
    if s is not None: #针对当个状态求解
        q=np.zeros(env.unwrapped.nA)
        for a in range(env.unwrapped.nA):
            for prob,next_state,reward,done in env.unwrapped.P[s][a]:
                q[a]+=prob*(reward+gamma*v[next_state]*(1.-done))
    else:  #针对所有状态求解
        q=np.zeros((env.unwrapped.nS,env.unwrapped.nA))
        for s in range(env.unwrapped.nS):
            q[s]=v2q(env,v,s,gamma)
    return q
def evaluate_policy(env,policy,gamma=1,tolerant=1e-6):
    v=np.zeros(env.unwrapped.nS)  #初始化状态价值函数
    num=0
    while True:
        num+=1
        #print(num)
        delta=0
        for s in range(env.unwrapped.nS):
            vs=sum(policy[s]*v2q(env,v,s,gamma))  #更新状态价值函数
            delta=max(delta,abs(v[s]-vs)) #更新最大误差
            v[s]=vs   #更新状态价值函数
        #print(delta)
        if delta<tolerant:  #查看是否满足迭代条件
            break
    return v

#对随机策略进行评估
print("状态价值函数：")
v_random=evaluate_policy(env,random_policy)
print(v_random.reshape((4,4)))
print("动作价值函数：")
q_random=v2q(env,v_random)
print(q_random)

def improv_policy(env,v,policy,gamma=1.):
    optimal=True
    for s in range(env.unwrapped.nS):
        q=v2q(env,v,s,gamma)
        a=np.argmax(q)
        if policy[s][a]!=1.:
            optimal=False
            policy[s]=0
            policy[s][a]=1.
    return optimal

policy=random_policy.copy()
optimal=improv_policy(env,v_random,policy)
if optimal:
    print("无更新，最优策略为：")
else:
    print("有更新策略为：")
print(policy)

#策略迭代的实现
def iterate_policy(env,gamma=1.,toterant=1e-6):
    policy=np.zeros((env.unwrapped.nS,env.unwrapped.nA))/env.unwrapped.nA #初始化为任意一个策略
    while True:
        v=evaluate_policy(env,policy,gamma,toterant) #策略评估
        if improv_policy(env,v,policy):
            break
    return policy,v

#利用迭代策略求最优策略
policy_pi,v_pi=iterate_policy(env)
print("状态价值函数：{}".format(v_pi.reshape(4,4)))
print("最优策略；{}".format(np.argmax(policy,axis=1).reshape(4,4)))

#价值迭代算法求解最优策略
def iterate_value(env,gamma,tolerant=1e-6):
    v=np.zeros(env.unwrapped.nS) #初始化
    while True:
        delta=0
        for s in range(env.unwrapped.nS):
            vmax=max(v2q(env,v,s,gamma))  #更新价值函数
            delta=max(delta,abs(v[s]-vmax))
            v[s]=vmax
        if delta<tolerant:
            break
        policy=np.zeros((env.unwrapped.nS,env.unwrapped.nA))  #计算最优策略
        for s in range(env.unwrapped.nS):
            a=np.argmax(v2q(env,v,s,gamma))
            policy[s][a]=1.
        return policy,v

policy_v1,v_vi=iterate_policy(env)
print("状态价值函数：{}".format(v_vi.reshape(4,4)))
print("最优策略：{}".format(np.argmax(policy_v1,axis=1).reshape(4,4)))
print("价值迭代 平均奖励：{}".format(play_policy(env,policy)))
