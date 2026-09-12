# td3_news_recommender.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import pickle
import os
from datetime import datetime

# ==================== 设备配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 1. 新闻环境 ====================
class NewsEnv:
    """新闻推荐仿真环境（基于特征与用户兴趣生成奖励）。"""

    def __init__(self, news_features_path):
        """
        初始化环境

        Args:
            news_features_path: 新闻特征文件路径
        """
        # 加载新闻特征
        with open(news_features_path, 'rb') as f:
            self.news_features = pickle.load(f)

        self.news_ids = list(self.news_features.keys())
        self.num_news = len(self.news_ids)

        # 新闻类别
        self.categories = ['社会', '财经', '科技', '娱乐', '体育',
                           '军事', '教育', '健康', '国际', '时尚']

        # 为每条新闻分配类别（从特征中获取）
        self.news_categories = {}
        for news_id in self.news_ids:
            self.news_categories[news_id] = self.news_features[news_id].get('category', '综合')

        # 生成用户（1000个）
        self.users = self._generate_users(1000)

        # 当前状态
        self.current_user = None
        self.view_history = []
        self.step_count = 0
        self.max_steps = 20

        # 特征维度
        self.feature_dim = len(next(iter(self.news_features.values()))['feature'])

        print("[环境] 初始化完成")
        print(f"   - 新闻数量: {self.num_news}")
        print(f"   - 特征维度: {self.feature_dim}")
        print(f"   - 用户数量: {len(self.users)}")

    def _generate_users(self, num_users):
        """生成用户"""
        users = []
        for i in range(num_users):
            # 每个用户对不同类别有不同兴趣
            interests = {cat: random.uniform(0, 1) for cat in self.categories}
            # 归一化
            total = sum(interests.values())
            interests = {cat: val / total for cat, val in interests.items()}

            users.append({
                'id': i,
                'interests': interests,
                'history': []
            })
        return users

    def reset(self):
        """重置环境"""
        self.current_user = random.choice(self.users)
        self.view_history = []
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """获取状态：用户兴趣 + 可用新闻统计"""
        # 用户兴趣向量 (10维)
        user_vec = np.array([self.current_user['interests'][cat]
                             for cat in self.categories])

        # 可用新闻的平均特征
        available = [nid for nid in self.news_ids if nid not in self.view_history]
        if available:
            features = np.stack([self.news_features[nid]['feature']
                                 for nid in available[:50]])
            news_mean = np.mean(features, axis=0)
        else:
            news_mean = np.zeros(self.feature_dim)

        # 状态拼接
        state = np.concatenate([user_vec, news_mean])
        return state

    def step(self, action):
        """
        执行动作
        action: 新闻ID（离散动作）
        """
        self.step_count += 1

        # 记录推荐
        self.view_history.append(action)
        self.current_user['history'].append(action)

        # 计算奖励
        reward = self._calculate_reward(action)

        # 下一个状态
        next_state = self._get_state()

        # 是否结束
        done = self.step_count >= self.max_steps

        return next_state, reward, done

    def _calculate_reward(self, news_id):
        if not self.current_user:
            return 0.0
        category = self.news_categories.get(news_id, '综合')
        interest = self.current_user['interests'].get(category, 0.0)
        # 根据兴趣值给予连续奖励，而不是二值
        if interest > 0.15:
            return 1.0 + (interest - 0.15) * 5  # 兴趣越高奖励越大，最高约 5.0
        elif interest > 0.05:
            return 0.2
        else:
            return -0.1  # 轻微惩罚

    def get_available_news(self):
        """获取可用新闻列表"""
        return [nid for nid in self.news_ids if nid not in self.view_history]


# ==================== 2. Actor 网络 ====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


# ==================== 3. Critic 网络 ====================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==================== 4. 经验回放 ====================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state = torch.FloatTensor(np.array([x[0] for x in batch])).to(device)
        action = torch.FloatTensor(np.array([x[1] for x in batch])).to(device)
        reward = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array([x[3] for x in batch])).to(device)
        done = torch.FloatTensor(np.array([x[4] for x in batch])).unsqueeze(1).to(device)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ==================== 5. TD3 智能体 ====================
class TD3Agent:
    def __init__(self, state_dim, action_dim, hidden_size=256,
                 actor_lr=3e-4, critic_lr=3e-3, gamma=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, buffer_size=100000, batch_size=64):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.total_steps = 0

        # 网络
        self.actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_size).to(device)

        self.target_actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim, hidden_size).to(device)

        # 初始化目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=critic_lr
        )

        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 损失函数
        self.mse_loss = nn.MSELoss()

        print("\n[TD3] 智能体已创建")
        print(f"   状态维度: {state_dim}")
        print(f"   动作维度: {action_dim}")
        print(f"   设备: {device}")

    def select_action(self, state, noise=0.1):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        if noise > 0:
            action = action + noise * np.random.randn(self.action_dim)

        return np.clip(action, -1, 1)

    def update(self):
        """TD3 更新"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # 采样
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # 为目标动作添加噪声
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor(next_state) + noise).clamp(-1, 1)

            # 计算目标Q值
            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        # 更新Critic
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        critic_loss1 = self.mse_loss(current_q1, target_q)
        critic_loss2 = self.mse_loss(current_q2, target_q)
        critic_loss = critic_loss1 + critic_loss2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新Actor
        actor_loss = None
        if self.total_steps % self.policy_delay == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_steps += 1
        return actor_loss.item() if actor_loss is not None else None, critic_loss.item()

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps
        }, path)
        print(f"[TD3] 模型已保存: {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_steps = checkpoint['total_steps']
        print(f"[TD3] 模型已加载: {path}")


# ==================== 6. 训练主程序 ====================
if __name__ == "__main__":
    import os

    # 配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    FEATURE_FILE = os.path.join(BASE_DIR, 'juhe_news_features.pkl')
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("TD3 新闻推荐 — 离线训练")
    print("=" * 60)

    # 1. 创建环境
    print("\n[环境] 创建中...")
    env = NewsEnv(FEATURE_FILE)

    # 2. 获取状态维度
    test_state = env.reset()
    state_dim = len(test_state)
    action_dim = 1

    # 3. 创建智能体
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=256,
        actor_lr=3e-4,
        critic_lr=3e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        buffer_size=100000,
        batch_size=64
    )

    # 4. 训练参数
    num_episodes = 200
    episode_rewards = []
    best_avg_reward = -float('inf')

    print("\n[训练] 开始...")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        # 噪声衰减
        if episode < 50:
            noise = 0.3
        else:
            noise = max(0.1 * (1 - (episode - 50) / (num_episodes - 50)), 0.05)

        while not done and step < 30:
            # 选择动作
            action = agent.select_action(state, noise=noise)

            # 将连续动作映射到新闻ID
            available = env.get_available_news()
            if not available:
                break

            idx = int((action[0] + 1) / 2 * (len(available) - 1))
            idx = max(0, min(idx, len(available) - 1))
            news_id = available[idx]

            # 执行动作
            next_state, reward, done = env.step(news_id)

            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 更新网络
            if len(agent.replay_buffer) > agent.batch_size:
                actor_loss, critic_loss = agent.update()

            state = next_state
            episode_reward += reward
            step += 1

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1:3d} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Noise: {noise:.3f}")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(os.path.join(MODEL_DIR, 'td3_new_best.pth'))
                print(f"  [检查点] 已保存最佳模型 (近10轮平均奖励: {avg_reward:.3f})")

    # 保存最终模型
    agent.save(os.path.join(MODEL_DIR, 'td3_new_final.pth'))

    print("\n" + "=" * 60)
    print("[训练] 已完成")
    print("=" * 60)
    print(f"\n训练统计:")
    print(f"   • 最佳平均奖励: {best_avg_reward:.3f}")
    print(f"   • 最终平均奖励: {np.mean(episode_rewards[-10:]):.3f}")