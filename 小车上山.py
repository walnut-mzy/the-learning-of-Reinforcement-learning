import gym
import numpy as np
env=gym.make("MountainCar-v0")
print("观测空间={}".format(env.observation_space))
print("动作空间={}".format(env.action_space))
print("观侧范围={}-{}".format(env.observation_space.low,env.observation_space.high))
print("动作数={}".format(env.action_space.n))
class BespokeAgent:
    def __init__(self,env):
        pass
    def decide(self,observation): #决策
        positon,velocity=observation
        lb=min(-0.09*(positon+0.25)**2+0.03,0.3*(positon+0.9)**4-0.008)
        ub=-0.07*(positon+0.38)**2+0.06
        if lb<velocity<ub:
            action=2
        else:
            action=0
        return action
    def learn(self,*args):
        pass
def play_montercarlo(env,agent,render=False,train=False):
    episode_reward=0   #记录回合总奖励，初始化为0
    observation=env.reset()  #重装游戏环境，开始新回合
    while True:
        if render: #判断是否显示
            env.render()
        action=agent.decide(observation)
        next_observation,reward,done,_=env.step(action)  #执行动作
        episode_reward+=reward   #收集回合奖励
        if train:   #判断石佛训练智能体
            agent.learn(observation,action,reward,done)   #学习
        if done:   #回合结束，跳出循环
            break
        observation=next_observation
    return episode_reward



agent=BespokeAgent(env)
env.seed(0)        #设计随机数种子，只是为了让结果可以精确复现，一般情况下可以删除
episode_reward=play_montercarlo(env,agent,render=True)
print("回合奖励={}".format(episode_reward))
env.close()

episode_rewards=[play_montercarlo(env,agent) for _ in range(100)]
print("平均回合奖励={}".format(np.mean(episode_rewards)))