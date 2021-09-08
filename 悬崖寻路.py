import gym
import numpy as np
import scipy
def play_once(env,policy):

    total_reward=0
    state=env.reset()
    loc=np.unravel_index(state,env.shape)
    print("状态={},位置={}".format(state,loc))
    while True:
        env.render()
        action=np.random.choice(env.nA,p=policy[state])
        next_state,reward,done,_=env.step(action)
        print("状态={},位置={},奖励={}".format(state,loc,reward))
        total_reward+=reward
        if done:
            break
        state=next_state
    return total_reward
def evaluate_bellman(env, policy, gamma=1.):
    a, b = np.eye(env.nS), np.zeros((env.nS))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            pi = policy[state][action]
            for p, next_state, reward, done in env.P[state][action]:
                a[state, next_state] -= (pi * gamma)
                b[state] += (pi * reward * p)
    v = np.linalg.solve(a, b)
    q = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for p, next_state, reward, done in env.P[state][action]:
                q[state][action] += ((reward + gamma * v[next_state]) * p)
    return v, q
def optimal_bellman(env, gamma=1.):
    p = np.zeros((env.nS, env.nA, env.nS))
    r = np.zeros((env.nS, env.nA))
    for state in range(env.nS - 1):
        for action in range(env.nA):
            for prob, next_state, reward, done in env.P[state][action]:
                p[state, action, next_state] += prob
                r[state, action] += (reward * prob)
    c = np.ones(env.nS)
    a_ub = gamma * p.reshape(-1, env.nS) - \
            np.repeat(np.eye(env.nS), env.nA, axis=0)
    b_ub = -r.reshape(-1)
    a_eq = np.zeros((0, env.nS))
    b_eq = np.zeros(0)
    bounds = [(None, None),] * env.nS
    res = scipy.optimize.linprog(c, a_ub, b_ub, bounds=bounds,
            method='interior-point')
    v = res.x
    q = r + gamma * np.dot(p, v)
    return v, q


env=gym.make("CliffWalking-v0")
print("观测空间={}".format(env.observation_space))
print("动作空间={}".format(env.action_space))
print("状态数量={},动作数量={}".format(env.nS,env.nA))
print("地图大小={}".format(env.shape))
# #最优策略
# actions=np.ones(env.shape,dtype=int)
# actions[-1,:]=0
# actions[:,-1]=2
# optimal_policy=np.eye(4)[actions.reshape(-1)]
# total_reward=play_once(env,optimal_policy)
# print("总奖励={}".format(total_reward))

optimal_state_values, optimal_action_values = optimal_bellman(env)
print('最优状态价值 = {}'.format(optimal_state_values))
print('最优动作价值 = {}'.format(optimal_action_values))
optimal_actions = optimal_action_values.argmax(axis=1)
print('最优策略 = {}'.format(optimal_actions))