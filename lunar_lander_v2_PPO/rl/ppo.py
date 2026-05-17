import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    # actor和critic共用1个网络，actor输出action_dim维度，critic输出1维度
    # 月球登陆器先只考虑离散动作空间，has_continuous_action_space=0
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            # 在当前状态s下，模型采取的策略
            # state[n, state_dim] -> actor -> action[n, action_dim]
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            # 激活函数
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            # 离散动作空间，使用softmax归一化，保证动作概率之和为1，稳定训练
                            nn.Softmax(dim=-1)
                        )
        # critic
        # 估计状态的价值函数V(s)，即当前状态s下，智能体未来能获得的期望回报
        # state[n, state_dim] -> critic -> value[n, 1]
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    # 离散动作空间不考虑
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    # 输入当前状态s，输出动作a，动作的概率，当前状态的价值
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            # 分类器，用来采样离散动作
            dist = Categorical(action_probs)

        # 基于动作空间的概率分布，采样得到离散动作分类
        action = dist.sample()
        action_logprob = dist.log_prob(action)  # logπθ，表示在状态s下，采取动作a的概率的对数
        state_val = self.critic(state)

        # 返回采样得到的动作，动作的概率，当前状态的价值
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    # 输入当前状态s和动作a，输出动作的概率，当前状态的价值，动作的熵
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)  # logπθ，表示在状态s下，采取动作a的概率的对数
        dist_entropy = dist.entropy()  # 动作的熵，熵越高，越倾向于采样概率低的动作，随机性高
        state_values = self.critic(state)
        
        # 返回动作的概率，当前状态的价值，动作的熵
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init  #初始化标准差。标准差决定了动作采样的 “随机性”

        # 在评估回报时，距离当前近的奖励影响越大，距离当前远的奖励影响越小，因此要引入折扣率discount rate γ
        self.gamma = gamma    #discount rate
        # off-policy特有，用于限制策略更新的幅度，防止策略更新过快
        self.eps_clip = eps_clip   #clip value
        self.K_epochs = K_epochs   #在1次策略更新中，权重参数被更新K_epochs次，表示搜集K_epochs次数据统一更新
        
        self.buffer = RolloutBuffer()   #init buffer to save state-action-reward

        # 只做学习的神经网络（lead agent）
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # 优化器，需要优化actor和critic的权重参数（同一套），但是它们学习率不同
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        # 只做策略采集的神经网络（subagent）
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    # 在当前状态s下，根据policy_old，采样得到动作a
    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    # 更新lead agent的策略
    def update(self):
        # Monte Carlo estimate of returns, 使用蒙特卡洛方法采样一个episode上的所有数据，从而估算每个t时间步的价值
        rewards = []
        discounted_reward = 0
        # 从当前最后一步开始，往前计算每个时间步的折扣回报
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0   # 多episode拼在一起时截断折扣回报，重新算这个episode的折扣回报
            discounted_reward = reward + (self.gamma * discounted_reward)   #贝尔曼最优公式
            rewards.insert(0, discounted_reward)
            
        # 经上面计算， rewards 表示第n步的价值v = 当前奖励 + 未来奖励的折扣
        # rewards[0] = reward[0] + gamma * rewards[1]
        # rewards[1] = reward[1] + gamma * rewards[2]
        # ...
        # rewards[n-1] = reward[n-1] + gamma * rewards[n]
        # rewards[n] = reward[n] + gamma * 0 = reward[n]

        # Normalizing the rewards，对价值做归一化
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages，A=Q-V
        # Q叫做动作价值，表示在状态s下，采取动作a的回报，来自于reward[n, 1]
        # V叫做状态价值，表示在状态s下，采取所有动作的平均回报，来自于critic网络[n, 1]
        # A=Q-V叫优势函数，表示在状态s下，采取动作a，比采取其他动作的回报高多少
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs，表示在1次更新中，更新K次策略
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            # r=exp(logπθ−logπθold)
            # 表示在状态s下，新策略（负责学习的agent）采取动作a的概率对数比上旧策略（负责采集策略的agent）采取动作a的概率对数
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            # advantages是旧策略的优势函数，计算后surr1是新策略的优势函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # Total loss = Policy loss + 0.5 * Value loss - 0.01 * Entropy loss
            # loss：表示采取当前策略的损失，越小越好
            # -torch.min(surr1, surr2)：让好的策略概率变大，让差的策略概率变小
            # 0.5 * self.MseLoss(state_values, rewards)：让采取策略后的状态价值v逼近真实回报，rewards就是上面的Q
            # - 0.01 * dist_entropy：让熵变大，鼓励策略别太确定，多探索
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


