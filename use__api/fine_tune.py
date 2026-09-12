# fine_tune.py
import json
import numpy as np
import torch
import os
from collections import defaultdict
from datetime import datetime
import copy

# 依赖同目录下的训练与 API 模块
from td3_news_recommender import TD3Agent, NewsEnv, device
from api_services_new import build_user_vector, CATEGORIES, news_cache

# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, 'user_history.jsonl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'td3_new_best.pth')
FINETUNED_MODEL_PATH = MODEL_PATH.replace('.pth', '_finetuned.pth')
FEATURE_FILE = os.path.join(BASE_DIR, 'juhe_news_features.pkl')
MIN_SAMPLES = 200          # 最少样本数
VALIDATION_RATIO = 0.2     # 验证集比例
FINETUNE_EPOCHS = 10
FINETUNE_UPDATES_PER_EPOCH = 50
BATCH_SIZE = 64
LEARNING_RATE_REDUCTION = 0.1  # 微调时降低学习率


def load_history():
    """加载历史推荐和反馈记录"""
    recommendations = []
    feedbacks = []
    if not os.path.exists(HISTORY_FILE):
        print(f"[微调] 历史文件不存在: {HISTORY_FILE}")
        return [], []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('type') == 'recommendation':
                    recommendations.append(data)
                elif data.get('type') == 'feedback':
                    feedbacks.append(data)
            except:
                continue
    print(f"[微调] 推荐记录 {len(recommendations)} 条, 反馈记录 {len(feedbacks)} 条")
    return recommendations, feedbacks


def compute_next_state(user_id, prev_state, feedback_news_id, reward):
    """
    根据反馈和奖励模拟下一状态。
    实际中，用户兴趣会根据奖励更新，新闻特征均值保持不变（候选池变化忽略）。
    这里仅更新用户兴趣向量部分，保持新闻特征均值部分不变。
    """
    # 解析 prev_state: 前10维是用户兴趣，后64维是新闻特征均值
    user_vec = prev_state[:10]
    news_mean = prev_state[10:]

    # 将 user_vec 转换为类别兴趣字典
    interests = {cat: float(user_vec[i]) for i, cat in enumerate(CATEGORIES)}

    # 获取反馈新闻的类别
    news_info = news_cache.get(feedback_news_id, {})
    category = news_info.get('category', '')
    if category in interests:
        # 应用与 /feedback 接口相同的更新逻辑
        interests[category] += reward * 0.05
        interests[category] = max(0.01, min(5.0, interests[category]))

    # 重新归一化用户兴趣向量（与 build_user_vector 一致）
    total = sum(interests.values())
    if total > 0:
        new_user_vec = [interests[cat] / total for cat in CATEGORIES]
    else:
        new_user_vec = [0.1] * 10

    # 拼接新的状态
    next_state = np.array(new_user_vec + news_mean.tolist(), dtype=np.float32)
    return next_state


def match_samples(recommendations, feedbacks):
    """
    将推荐和反馈匹配为 (state, action, reward, next_state, done) 样本。
    策略：对于每个推荐请求，寻找该用户在此推荐之后第一个对该推荐列表中新闻的反馈。
    """
    # 按用户分组
    user_recs = defaultdict(list)
    for rec in recommendations:
        user_recs[rec['user_id']].append(rec)
    user_fbs = defaultdict(list)
    for fb in feedbacks:
        user_fbs[fb['user_id']].append(fb)

    samples = []
    for user_id in user_recs:
        if user_id not in user_fbs:
            continue

        recs = sorted(user_recs[user_id], key=lambda x: x['timestamp'])
        fbs = sorted(user_fbs[user_id], key=lambda x: x['timestamp'])

        fb_idx = 0
        for rec in recs:
            rec_time = rec['timestamp']
            rec_news = rec.get('recommended_news', [])
            actions = rec.get('actions', [])
            if not actions:
                continue
            action = actions[0] if len(actions) == 1 else actions[0]  # 取第一个动作值（实际推荐时每个推荐一个动作）
            state = np.array(rec['state'], dtype=np.float32)

            # 寻找该推荐之后最早的反馈（且新闻在推荐列表中）
            found = None
            while fb_idx < len(fbs) and fbs[fb_idx]['timestamp'] >= rec_time:
                fb = fbs[fb_idx]
                if fb['news_id'] in rec_news:
                    found = fb
                    fb_idx += 1
                    break
                fb_idx += 1

            if found is not None:
                reward = found['reward']
                # 计算下一状态（需要知道反馈新闻ID）
                next_state = compute_next_state(user_id, state, found['news_id'], reward)
                done = False
                samples.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })
    print(f"[微调] 匹配得到训练样本 {len(samples)} 条")
    return samples


def evaluate(agent, env, num_episodes=5):
    """评估当前策略的平均奖励（使用模拟环境）"""
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for _ in range(10):  # 每个 episode 最多10步
            action = agent.select_action(state, noise=0)
            # 将动作映射为新闻ID
            available = env.get_available_news()
            if not available:
                break
            idx = int((action[0] + 1) / 2 * (len(available) - 1))
            idx = max(0, min(idx, len(available) - 1))
            news_id = available[idx]
            next_state, reward, done = env.step(news_id)
            episode_reward += reward
            state = next_state
            if done:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


def fine_tune():
    print("=" * 60)
    print("TD3 模型 — 在线反馈微调")
    print("=" * 60)

    # 1. 加载历史数据
    recommendations, feedbacks = load_history()
    samples = match_samples(recommendations, feedbacks)
    if len(samples) < MIN_SAMPLES:
        print(f"[微调] 样本不足（需要 {MIN_SAMPLES} 条，当前 {len(samples)}），已中止")
        return

    # 2. 划分训练集和验证集
    np.random.shuffle(samples)
    split = int(len(samples) * (1 - VALIDATION_RATIO))
    train_samples = samples[:split]
    val_samples = samples[split:]
    print(f"[微调] 训练集 {len(train_samples)} 条, 验证集 {len(val_samples)} 条")

    # 3. 初始化环境并获取状态维度
    env = NewsEnv(FEATURE_FILE)
    test_state = env.reset()
    state_dim = len(test_state)

    # 4. 加载预训练模型
    agent = TD3Agent(state_dim=state_dim, action_dim=1, hidden_size=256)
    agent.load(MODEL_PATH)

    # 降低学习率进行微调
    for param_group in agent.actor_optimizer.param_groups:
        param_group['lr'] *= LEARNING_RATE_REDUCTION
    for param_group in agent.critic_optimizer.param_groups:
        param_group['lr'] *= LEARNING_RATE_REDUCTION
    print(f"[微调] 学习率已乘以 {LEARNING_RATE_REDUCTION}")

    # 5. 将训练样本填入经验回放缓冲区
    for sample in train_samples:
        agent.replay_buffer.push(
            sample['state'],
            sample['action'],
            sample['reward'],
            sample['next_state'],
            sample['done']
        )
    print(f"[微调] 已向经验回放写入 {len(train_samples)} 条，缓冲区大小 {len(agent.replay_buffer)}")

    # 6. 评估微调前性能
    print("\n[微调] 评估微调前策略...")
    pre_reward = evaluate(agent, env)
    print(f"微调前平均奖励: {pre_reward:.2f}")

    # 7. 微调
    best_val_reward = -float('inf')
    best_model_state = None

    for epoch in range(FINETUNE_EPOCHS):
        # 每轮将训练样本再次混洗并重新加入缓冲区（可选）
        np.random.shuffle(train_samples)
        for _ in range(FINETUNE_UPDATES_PER_EPOCH):
            # 随机采样一个训练样本（也可以使用缓冲区采样，这里为了简单直接使用样本）
            # 实际应使用 agent.replay_buffer.sample，但当前缓冲区中已有所有训练样本
            if len(agent.replay_buffer) >= BATCH_SIZE:
                actor_loss, critic_loss = agent.update()
        # 验证
        val_reward = evaluate(agent, env)
        print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS}, 验证平均奖励: {val_reward:.2f}")
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            best_model_state = copy.deepcopy(agent.actor.state_dict())
            print(f"  → 新最佳模型 (奖励 {val_reward:.2f})")

    # 8. 恢复最佳模型
    if best_model_state is not None:
        agent.actor.load_state_dict(best_model_state)
        print(f"\n[微调] 最佳验证平均奖励: {best_val_reward:.2f}")

    # 9. 保存微调后的模型
    agent.save(FINETUNED_MODEL_PATH)
    print(f"[微调] 模型已保存: {FINETUNED_MODEL_PATH}")

    # 10. 评估最终性能
    final_reward = evaluate(agent, env)
    print(f"\n[微调] 微调后平均奖励: {final_reward:.2f}")
    improvement = final_reward - pre_reward
    if improvement > 0:
        print(f"[微调] 相对微调前提升: +{improvement:.2f}")
    else:
        print(f"[微调] 相对微调前未提升 ({improvement:.2f})，可继续使用原 checkpoint")


if __name__ == "__main__":
    fine_tune()