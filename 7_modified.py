
import time
# from based import *
from common import *
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


from torch.distributions import Categorical
from plshown import Visualization
import pandas as pd
import logging

logging.basicConfig(encoding='utf-8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK_COMPLETION_REWARD = 100

def get_load_data():
    # 读取 HDF5 文件
    with pd.HDFStore('D:\paper\code\muldp\second\data\data.h5', mode='r') as store:
        # 读取 location_df
        location_df = store.get('df/location_df')

        # 读取 crane_df
        crane_df = store.get('df/crane_df')

        # 读取 ladle_data
        ladle_data = store.get('df/ladle_data')

        # 读取 hangjob_df
        hangjob_df = store.get('df/hangjob_df')

    return [location_df, crane_df, ladle_data, hangjob_df]



class State:  # 定义状态类
    def __init__(self, state_size):
        self.crane_state = state_size

    def set_state(self, state_size):
        self.crane_state = state_size

    def get_state(self):  # 获取状态的函数
        return self.state_size  # 返回当前状态


class Action:  # 定义动作类
    def __init__(self, action_size):
        self.crane_action = action_size

    def set_action(self, action_size):
        self.crane_action = action_size

    def get_action(self):  # 获取动作的函数
        return self.action_size  # 返回当前动作


class Environment:  # 定义环境类
    def __init__(self, data):  # 初始化函数，参数为状态和动作的尺寸
        self.data = data
        [self.location_df, self.crane_df, self.ladle_data, self.hangjob_df] = data

        self.crane_states = []
        self.casting_states = []
        # 获取连铸机列表
        self.casting_rows = self.location_df[self.location_df['loc_location'].str.contains('CC')]

        self.span3_range, self.span4_range = span_pox_range(data) #过跨线3  和  过跨线4  的位置范围
        self.timeline = [] #开始时间，结束时间，开始位置，结束位置

        self.state_size = self.get_state_size()  # 通过数据的第一行的长度来初始化状态的尺寸
        self.state = State(self.state_size)  # 创建状态对象
        self.init_time = -600 #设置场景时间

        self.reward = 0  # 用于记录当前回合的奖励
        self.state = self.reset()
        # 其他初始化代码
        self.total_reward = 0
        self.crane_events = []
        self.task_events = []

        # 初始化环境变量
        self.n_crane = len(self.crane_df)  # 行车的数量

        #惩罚
        self.COLLISION_PENALTY = -10000
        #安全距离
        self.SAFETY_DISTANCE = 100

        #失败任务数量
        self.threshold = 5  # 设置阈值，根据你的实际需求调整

        self.completion_rate_threshold = 0.8

        self.reward_params = {
            'select_task': 10,  # 选择任务的奖励
            'no_task': -10,  # 没有任务时的惩罚
            'pick_up_ladle': 20,  # 拾取铁水罐的奖励
            'wrong_action': -20,  # 执行错误动作的惩罚
            'reach_target': 30,  # 到达目标位置的奖励
            'conflict': -300,  # 发生冲突的惩罚
            'put_down_ladle': 40,  # 放下铁水罐的奖励
            'late_task': -50,  # 任务延迟的惩罚
            'casting_interruption': -300,  # 连铸机中断的惩罚
            'casting_time_threshold': 60,  # 连铸机中断时间阈值
            'no_task_negative': -10,  # 无任务时，如果行车进行了其他动作的惩罚
            'no_task_position': -10,  # 无任务时，如果行车静止的奖励
            'false_task': -300,  # 任务执行失败的惩罚
            'true_task': 100,  # 任务执行成功的奖励
            'going_task': 50,  # 正在执行任务的奖励
        }

    def get_state_size(self):
        """
        获取环境状态的大小。

        环境状态由行车状态和钢包状态组成。
        行车状态：编号，位置（pos_x）、是否有任务（在MOD_CRANE_HANGJOB表中是否存在对应的job_id）、任务起始位置（start_loc）、任务目标位置（dest_loc）、是否正在抓取钢包（crane_status中是否表示正在抓取） (共5个状态变量)。
        钢包状态：是否被占用（status字段表示） (共1个状态变量)。

        因此，总的状态空间的大小就是 (行车数量 * 行车状态变量数量) + (钢包数量 * 钢包状态变量数量)。
        """
        # num_ladles = len(self.location_df[self.location_df['loc_location'].str.contains('KB')])
        # return len(self.crane_df) * 5 + num_ladles

        return len(self.crane_df) * 6 + len(self.casting_rows) * 4

    def get_action_size(self):
        """
        获取行动空间的大小。

        行动由两部分组成：行车编号、行车动作。

        行车编号：行车的数量决定了行车编号的可能性。
        行车动作：每个行车都有4种可能的动作，包括向左移动、向右移动、提起钢包、放下钢包。

        所以总的动作空间的大小就是 行车数量 * 行车动作数量
        """
        num_crane_actions = 5  # Move left, move right, lift, drop
        return num_crane_actions
        # return len(self.crane_df) * num_crane_actions

    def reset(self):
        """
        重置环境的状态，初始化行车的状态和钢包的状态。

        行车状态：位置初始化为init_pos_x，没有任务，任务起始位置和任务目标位置为None，没有抓取钢包。
        钢包状态：所有的钢包初始化为未被占用。

        返回：环境的初始状态
        """
        # Initialize crane states
        self.crane_states = [] #初始化为空
        for index, row in self.crane_df.iterrows():
            crane_state = {
                "crane_name": index, #行车id
                "pre_location": 0, #行车上一个事件的位置
                "cur_location": row["init_pos_x"], #行车当前事件的位置
                "ladle_id":0, #钢包id
                "cur_time": self.init_time, #当前时间
                "task_id":0, #执行任务的id

            }
            self.crane_states.append(crane_state)

        # 初始化连铸机状态字典

        self.casting_states = [] # 初始化为空
        for index, casting_machine in self.casting_rows.iterrows():
            casting_state = {
                'casting_name': index, #当前的连铸机名称
                'cur_time': 0, #当前的时间
                'cur_state': 0,  # 初始状态为断浇
                'finish_time': 0, # 到该时间点断浇
            }
            self.casting_states.append(casting_state)

        # 将行车状态和连铸机状态列表转化为数组，然后拼接起来
        crane_to_row = np.array(self.crane_states).flatten()
        casting_to_row = np.array(self.casting_states).flatten()
        state_array = np.concatenate([crane_to_row, casting_to_row])

        return state_array


    def find_nearest_tilt_table(self, current_position, tilt_table_df):
        tilt_table_df.loc[:, 'distance'] = abs(tilt_table_df['pos_x'] - current_position)

        nearest_tilt_table = tilt_table_df.loc[tilt_table_df['distance'].idxmin()]
        return nearest_tilt_table

    def find_nearest_location(self, current_pos, target_keyword):
        # 找出包含目标关键字的所有位置
        matching_locations = self.location_df[self.location_df['loc_location'].str.contains(target_keyword)].copy()

        # 计算这些位置与当前位置的距离
        # matching_locations['distance'] = abs(matching_locations['pos_x'] - current_pos)
        matching_locations.loc[:, 'distance'] = abs(matching_locations['pos_x'] - current_pos)

        # 返回距离最近的位置
        return matching_locations.loc[matching_locations['distance'].idxmin()]


    # def step(self, actions):
    #     """
    #     输入：动作 action
    #     功能：执行动作，更新环境状态，返回新的状态和奖励
    #     输出：新的状态，奖励，是否结束
    #     """
    #
    #     # 初始化奖励为 0
    #     reward = 0
    #     # 对于每一个行车
    #     for i, crane_state in enumerate(self.crane_states):
    #         action = actions[i]
    #         # 根据动作更新行车状态，并得到奖励
    #         self.crane_states[i], self.casting_states, self.timeline, reward, self.hangjob_df = update_crane_state(self.data, self.casting_rows, self.casting_states, crane_state, action, self.hangjob_df, self.timeline, self.reward_params)
    #
    #     # # 检查任务的完成状态，并根据结果给予奖励或惩罚
    #     # max_cur_time = max(crane['cur_time'] for crane in self.crane_states)
    #     # reward += check_task_completion(max_cur_time, self.hangjob_df, self.reward_params)
    #
    #     # 收集所有智能体的状态
    #     next_crane_states = []
    #     for crane_state in self.crane_states:
    #         next_crane_states.append(crane_state.copy())
    #
    #     # 收集所有连铸机的状态
    #     next_casting_states = []
    #     for casting_state in self.casting_states:
    #         next_casting_states.append(casting_state.copy())
    #
    #     # # 将行车状态和连铸机状态列表转化为数组，然后拼接起来
    #     # crane_to_row = np.array(self.crane_states).flatten()
    #     # casting_to_row = np.array(self.casting_states).flatten()
    #     #
    #     next_states = np.concatenate([np.array(self.crane_states).flatten(), np.array(self.casting_states).flatten()])
    #
    #
    #     # 判断是否完成所有任务
    #     unfinished_tasks = [task for task in self.hangjob_df.itertuples(index=False) if task.status == 0]
    #     done = len(unfinished_tasks) <= self.threshold
    #
    #
    #
    #     return next_states, next_crane_states, next_casting_states, reward, done

    def step(self, actions):
        """
        输入：动作 action
        功能：执行动作，更新环境状态，返回新的状态和奖励
        输出：新的状态，新的行车状态，新的连铸状态，奖励，是否结束
        """

        # 初始化奖励为 0
        rewards = [0 for _ in self.crane_states]

        # 对于每一个行车，尝试执行动作并更新状态
        next_crane_states = []
        next_casting_states = []
        for i, crane_state in enumerate(self.crane_states):
            # print("***************************crane_state['cur_location']********************",crane_state['cur_location'])
            action = actions[i]
            next_crane_state, next_casting_state, next_timeline, reward, next_hangjob_df = update_crane_state(
                self.data, self.casting_rows, self.casting_states, crane_state, action, self.hangjob_df, self.timeline,
                self.reward_params)
            next_crane_states.append(next_crane_state)
            next_casting_states = next_casting_state
            self.timeline = next_timeline  # 添加新的时间线到环境的时间线
            rewards[i] += reward

        # 检查是否存在行车碰撞，若存在，给与惩罚
        collision_count = calculate_crane_collisions(next_timeline)
        if collision_count > 0:
            for i in range(len(rewards)):
                rewards[i] += self.reward_params['conflict'] * collision_count

        # 检查是否存在连铸机中断，若存在，给与惩罚
        interruption_count = calculate_casting_interruptions(next_casting_states, next_timeline)
        if interruption_count > 0:
            for i in range(len(rewards)):
                rewards[i] += self.reward_params['casting_interruption'] * interruption_count

        # 检查任务完成率，若过低，给与惩罚
        completion_rate = calculate_task_completion_rate(next_hangjob_df)
        if completion_rate < self.completion_rate_threshold:
            for i in range(len(rewards)):
                rewards[i] += self.reward_params['false_task']

        # 更新状态
        self.crane_states = next_crane_states
        self.casting_states = next_casting_states
        self.timeline = next_timeline
        self.hangjob_df = next_hangjob_df

        # 判断是否完成所有任务
        unfinished_tasks = [task for task in self.hangjob_df.itertuples(index=False) if task.status == 0]
        done = len(unfinished_tasks) <= self.threshold

        # 将状态转化为数组
        next_states = np.concatenate([np.array(self.crane_states).flatten(), np.array(self.casting_states).flatten()])

        # 将奖励列表转化为总奖励
        reward = sum(rewards)

        return next_states, next_crane_states, next_casting_states, reward, done

    def get_total_reward(self):
        return self.total_reward

    def get_crane_events(self):
        return self.crane_events

    def get_task_events(self):
        return self.task_events


# 定义Actor网络
class Actor(nn.Module):
    # 初始化Actor网络
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        # 定义一个简单的全连接网络，它将状态映射到动作空间的概率分布
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),  # 第一层：线性层，将状态映射到128个节点
            nn.ReLU(),  # 第二层：ReLU激活函数
            nn.Linear(128, action_size),  # 第三层：线性层，将128个节点映射到动作空间大小
            nn.Softmax(dim=-1)  # 第四层：Softmax激活函数，将输出转化为概率分布
        )

    # 定义前向传播的过程
    def forward(self, state):
        return self.network(state)

# 定义Critic网络
class Critic(nn.Module):
    # 初始化Critic网络
    def __init__(self, state_size):
        super(Critic, self).__init__()
        # 定义一个简单的全连接网络，它将状态映射到一个标量值（即该状态的预期回报）
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),  # 第一层：线性层，将状态映射到128个节点
            nn.ReLU(),  # 第二层：ReLU激活函数
            nn.Linear(128, 1)  # 第三层：线性层，将128个节点映射到1个节点
        )

    # 定义前向传播的过程
    def forward(self, state):
        print("state",state)
        return self.network(state)

# 定义PPO算法的经验回放缓冲区
class PPOBuffer:
    def __init__(self):
        # 初始化各个列表
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    # 定义存储经验的方法
    def store(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    # 定义获取所有经验的方法
    def get(self):
        return self.states, self.actions, self.rewards, self.next_states, self.dones, self.log_probs

    # 定义清空经验的方法
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()

# 定义多智能体类
class MultiAgent:
    # 初始化多智能体类
    def __init__(self, state_size, action_size, n_agents, epsilon=1.0, min_epsilon=0.01, high_reward_threshold=100.0):
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.buffer = PPOBuffer()  # 创建PPO缓冲区
        self.memory = deque(maxlen=200000)  # 你可以自己设置最大长度

        self.actor = Actor(state_size, action_size).to(device)  # 创建Actor网络
        self.critic = Critic(state_size).to(device)  # 创建Critic网络

        self.optimizer_actor = optim.Adam(self.actor.parameters())  # 创建Actor的优化器
        self.optimizer_critic = optim.Adam(self.critic.parameters())  # 创建Critic的优化器

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.high_reward_threshold = high_reward_threshold

    def update_epsilon(self, reward):
        """
        Decreases epsilon more rapidly when the agent is doing well (i.e., when it's receiving high rewards).
        """
        if reward > self.high_reward_threshold:
            decay_rate = 0.99
        else:
            decay_rate = 0.999

        self.epsilon = max(self.epsilon * decay_rate, self.min_epsilon)

    # 定义如何根据状态选择动作的方法
    def act(self, state):
        state_tensor = torch.from_numpy(state[0].astype(np.float32)).to(device)
        action_probs = self.actor(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        action_mapping = {-2: 0, -1: 1, 0:2, 1: 3, 2: 4}
        inverse_mapping = {v: k for k, v in action_mapping.items()}
        return inverse_mapping[action.item()], log_prob.item()

    def step(self, state, action, reward, next_state, done, log_prob):
        self.buffer.store(state, action, reward, next_state, done, log_prob)  # 存储经验到缓冲区

    # 定义如何学习的方法
    def learn(self, discount_factor=0.99, clip_epsilon=0.2):
        states, actions, rewards, next_states, dones, old_log_probs = self.buffer.get()  # 获取所有经验

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)

        # 计算预期回报
        print("next_states",next_states)
        expected_returns = rewards + discount_factor * self.critic(next_states) * (1 - dones)

        # 计算Critic的损失函数
        critic_loss = nn.MSELoss()(self.critic(states), expected_returns.detach())

        # 计算Actor的损失函数（使用PPO的clip技巧）
        action_probs = self.actor(states)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate1 = ratio * expected_returns
        surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * expected_returns
        actor_loss = -torch.min(surrogate1, surrogate2)

        # 进行反向传播和优化
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.buffer.clear()  # 清空缓冲区

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 定义如何保存模型的方法
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
        }, path)

    # 定义如何加载模型的方法
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])


def train_model(divide_data, model_save_path):
    print("train_model_start")
    EPISODES = 2
    batch_size = 2
    visualize_interval = 2
    # 初始化历史记录
    crane_states_history = []
    casting_states_history = []

    # Initialize environment
    env = Environment(divide_data)

    state_size = env.get_state_size()
    action_size = env.get_action_size()

    # Create a list of agents
    agents = [MultiAgent(state_size, action_size, len(env.crane_df)) for _ in range(len(env.crane_df))]

    # Assuming that each agent has its own weights
    for agent_idx, agent in enumerate(agents):
        try:
            agent.load(f"{model_save_path}model_weights_{agent_idx}.pt")
        except:
            print(f"No previous model weights found for agent {agent_idx}, starting training from scratch.")

    for e in range(EPISODES):
        # Reset the environment
        state = env.reset()
        done = False
        step_count = 0
        while not done:
            step_count += 1
            actions = []
            log_probs = []  # 初始化 log_probs
            state_flatten = [value for dict_ in state for value in dict_.values()]
            state_array = np.array(state_flatten).reshape(1, -1)
            for agent_idx, agent in enumerate(agents):
                # Get action for each agent
                action, log_prob = agent.act(state_array)  # Pass the entire environment's state to the act method
                actions.append(action)
                log_probs.append(log_prob)

            next_state, next_crane_states, casting_states, reward, done = env.step(actions)

            # 让每个智能体记住全局状态，而不是只记住自己的部分
            for agent_idx, agent in enumerate(agents):
                agent.step(state, actions[agent_idx], reward, next_state, done, log_probs[agent_idx])  # 存储经验

            state = next_state

            if done:
                print(f"Episode: {e}/{EPISODES}")
                for agent in agents:
                    if len(agent.memory) > batch_size:
                        agent.learn()

            # 在每一步后，将新的状态添加到历史记录中
            # crane_states_history.append(next_crane_states.copy())
            # casting_states_history.append(casting_states.copy())

            if step_count == 500:
                break

        # Save model weights for each agent every 20 episodes
        if e % 2 == 0:
            for agent_idx, agent in enumerate(agents):
                agent.save(f"{model_save_path}model_weights_{agent_idx}.pt")

        # Update epsilon for each agent after each episode
        total_reward = env.get_total_reward()
        for agent in agents:
            agent.update_epsilon(total_reward)

        # 任务完成率、行车碰撞次数、钢水断浇次数
        # 使用方式：
        # env 是你的环境对象
        collision_count = calculate_crane_collisions(env.timeline)  # 计算行车碰撞次数
        interruption_count = calculate_casting_interruptions(env.casting_states, env.timeline)  # 计算连铸机断浇次数
        completion_rate = calculate_task_completion_rate(env.hangjob_df)  # 计算任务完成率

        print("行车碰撞次数:", collision_count)  # 打印行车碰撞次数
        print("连铸机断浇次数:", interruption_count)  # 打印连铸机断浇次数
        print("任务完成率:", completion_rate)  # 打印任务完成率


    print("train_model_end")


def test_model(divide_data,model_save_path):
    print("test_model_start")
    env = Environment(divide_data)
    state_size = env.get_state_size()
    action_size = env.get_action_size()

    # 初始化历史记录
    crane_states_history = []
    casting_states_history = []

    # Create a list of agents
    agents = [MultiAgent(state_size, action_size, len(env.crane_df)) for _ in range(len(env.crane_df))]

    # Assuming that each agent has its own weights
    for agent_idx, agent in enumerate(agents):
        try:
            agent.load(f"{model_save_path}model_weights_{agent_idx}.pt") # 根据 MultiAgent 类中的 save 方法，应该加载的是 .pt 文件，而不是 .h5 文件
        except:
            print(f"No previous model weights found for agent {agent_idx}, starting training from scratch.")

    state = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    while not done:
        step_count += 1
        state_flatten = [value for dict_ in state for value in dict_.values()]
        state_array = np.array(state_flatten).reshape(1, -1)
        actions = [agent.act(state_array) for agent in agents] # 无需先将状态 flatten 和 reshape，因为在 MultiAgent 的 act 方法中，状态会被转换为张量
        next_state, next_crane_states, casting_states, reward, done = env.step(actions)
        total_reward += reward
        state = next_state # 直接用 next_state 更新 state，无需先 flatten 和 reshape

        # 在每一步后，将新的状态添加到历史记录中
        crane_states_history.append(next_crane_states.copy())
        casting_states_history.append(casting_states.copy())

        if step_count == 100:
            break
        if done:
            print("Test episode finished.")
            break

    print("total_reward", total_reward)

    # 任务完成率、行车碰撞次数、钢水断浇次数
    # 使用方式：
    # env 是你的环境对象
    collision_count = calculate_crane_collisions(env.timeline)  # 计算行车碰撞次数
    interruption_count = calculate_casting_interruptions(env.casting_states, env.timeline)  # 计算连铸机断浇次数
    completion_rate = calculate_task_completion_rate(env.hangjob_df)  # 计算任务完成率

    print("行车碰撞次数:", collision_count)  # 打印行车碰撞次数
    print("连铸机断浇次数:", interruption_count)  # 打印连铸机断浇次数
    print("任务完成率:", completion_rate)  # 打印任务完成率

    # 设置日志级别为 INFO，日志文件名为 my_log.log，文件打开模式为 'w'（写入模式，如果文件已存在则清空）
    logging.basicConfig(filename='my_log.log', level=logging.INFO, filemode='w')

    # 使用 logging.info() 记录信息
    logging.info("行车碰撞次数: %s", collision_count)
    logging.info("连铸机断浇次数: %s", interruption_count)
    logging.info("任务完成率: %s", completion_rate)

    print("crane_states_history:",crane_states_history)
    print("casting_states_history:",casting_states_history)

    # 创建一个 Visualization 对象
    vis = Visualization()
    # 获取行车的状态信息
    crane_states = [crane_state.copy() for crane_state in crane_states_history]
    # 获取连铸机的状态信息
    casting_states = [casting_state.copy() for casting_state in casting_states_history]

    # 行车轨迹
    vis.plot_crane_trajectory(crane_states)
    # 连铸机状态
    vis.plot_casting_state(casting_states)
    # 显示图像
    vis.show()




def main():
    group_size = 20
    data = get_load_data()
    cross_lines = get_cross_lines(data[0])

    tasklist = divide_tasks(group_size,data[3])
    model_save_path = "./cache/"
    for task in tasklist:
        divide_data = [data[0],data[1],data[2],task]
        train_model(divide_data,model_save_path)
    start_time = time.time()
    test_model(data,model_save_path)
    end_time = time.time()
    print(f"Testing execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
