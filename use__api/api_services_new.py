# api_services_new.py
import sys
import os
import time
import json
import pickle
import numpy as np
import random
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import requests
from bs4 import BeautifulSoup
import pymysql
import cloudscraper
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import quote
from flask import send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
import math

# ==================== TD3 Actor（与训练 checkpoint 结构一致，服务仅推理）====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class TD3Agent:
    """在线推理用：仅含 Actor，从训练保存的 checkpoint 加载权重。"""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, hidden_size).to(self.device)
        self.actor.eval()

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        print(f"[模型] 已加载: {path}")

    def select_action(self, state, noise=0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        return action


# ==================== 基础路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FEATURE_FILE = os.path.join(BASE_DIR, 'juhe_news_features.pkl')
FEEDBACK_FILE = os.path.join(BASE_DIR, 'feedback_data.jsonl')
USER_HISTORY_FILE = os.path.join(BASE_DIR, 'user_history.jsonl')
USERS_FILE = os.path.join(BASE_DIR, 'users.json')
USER_STATE_FILE = os.path.join(BASE_DIR, 'user_state.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'td3_new_best.pth')
CATEGORIES = ['社会', '财经', '科技', '娱乐', '体育', '军事', '教育', '健康', '国际', '时尚']

print("=" * 60)
print("新闻推荐系统 API (TD3)")
print("=" * 60)
print(f"模型目录: {MODEL_DIR}")
print(f"特征文件: {FEATURE_FILE}")
print(f"反馈文件: {FEEDBACK_FILE}")
print(f"用户历史: {USER_HISTORY_FILE}")
print(f"用户文件: {USERS_FILE}")
print(f"用户状态: {USER_STATE_FILE}")

# ==================== MySQL 配置 ====================
# 按你的实际环境修改这几个配置即可
MYSQL_CONFIG = {
    'host': os.environ.get('NEWS_MYSQL_HOST', '127.0.0.1'),
    'port': int(os.environ.get('NEWS_MYSQL_PORT', '3306')),
    'user': os.environ.get('NEWS_MYSQL_USER', 'root'),
    'password': os.environ.get('NEWS_MYSQL_PASSWORD', '123456'),
    'database': os.environ.get('NEWS_MYSQL_DB', 'news_recommender'),
    'charset': 'utf8mb4'
}


def get_mysql_conn():
    try:
        conn = pymysql.connect(
            host=MYSQL_CONFIG['host'],
            port=MYSQL_CONFIG['port'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            database=MYSQL_CONFIG['database'],
            charset=MYSQL_CONFIG['charset'],
            autocommit=True,
        )
        return conn
    except Exception as e:
        print(f"[MySQL] 连接失败，已降级为本地 JSON: {e}")
        return None


def init_mysql():
    """初始化 MySQL 数据库表（如果存在则跳过）"""
    conn = get_mysql_conn()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            # 用户表（存画像）
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    username VARCHAR(191) NOT NULL UNIQUE,
                    interests_json JSON NOT NULL,
                    last_login DATETIME NULL,
                    PRIMARY KEY (id),
                    KEY idx_users_username (username)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            # 反馈表
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    user_id VARCHAR(191) NOT NULL,
                    news_id VARCHAR(191) NOT NULL,
                    clicked TINYINT(1) NOT NULL DEFAULT 0,
                    dwell_time INT NOT NULL DEFAULT 0,
                    liked TINYINT(1) NOT NULL DEFAULT 0,
                    reward DOUBLE NOT NULL DEFAULT 0,
                    timestamp DATETIME NOT NULL,
                    PRIMARY KEY (id),
                    KEY idx_feedback_user (user_id),
                    KEY idx_feedback_news (news_id),
                    KEY idx_feedback_time (timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            # 收藏表
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_favorites (
                    user_id VARCHAR(191) NOT NULL,
                    news_id VARCHAR(191) NOT NULL,
                    created_at DATETIME NOT NULL,
                    PRIMARY KEY (user_id, news_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            # 已读表
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_read_history (
                    user_id VARCHAR(191) NOT NULL,
                    news_id VARCHAR(191) NOT NULL,
                    read_count INT NOT NULL DEFAULT 0,
                    last_read_ts BIGINT NOT NULL,
                    PRIMARY KEY (user_id, news_id),
                    KEY idx_read_user (user_id),
                    KEY idx_read_news (news_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            # 推荐曝光表
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_exposure (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    request_id VARCHAR(191) NOT NULL,
                    user_id VARCHAR(191) NOT NULL,
                    news_id VARCHAR(191) NOT NULL,
                    rank_pos INT NOT NULL,
                    score DOUBLE NOT NULL DEFAULT 0,
                    category VARCHAR(64) NULL,
                    source VARCHAR(191) NULL,
                    exposed_at DATETIME NOT NULL,
                    PRIMARY KEY (id),
                    KEY idx_exposure_user (user_id),
                    KEY idx_exposure_news (news_id),
                    KEY idx_exposure_time (exposed_at),
                    KEY idx_exposure_request (request_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
        print("[MySQL] 表结构检查/创建完成")
    finally:
        conn.close()

# ==================== 创建必要的目录 ====================
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 全局变量 ====================
app = Flask(__name__)
CORS(app)
agent = None
news_cache = {}
model_loaded = False

# ==================== /article 缓存与会话 ====================
article_content_cache = {}  # {news_id: {"ts": float, "data": {...}}}
article_content_cache_lock = threading.Lock()
ARTICLE_CACHE_TTL_SECONDS = 3600
ARTICLE_CACHE_MAX_ITEMS = 200
def create_scraper_session():
    # 使用 cloudscraper 绕过 Cloudflare
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'mobile': False
        }
    )
    # 设置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    scraper.mount("http://", adapter)
    scraper.mount("https://", adapter)
    return scraper

article_session = create_scraper_session()

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
]

def get_headers(url):
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://www.google.com/',  # 部分站点校验 Referer，降低拒答概率
    }

retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
article_session.mount('http://', adapter)
article_session.mount('https://', adapter)
time.sleep(random.uniform(1, 3))


# ==================== 加载新闻特征函数 ====================
def load_news_features():
    """加载新闻特征缓存"""
    global news_cache, CATEGORIES
    if os.path.exists(FEATURE_FILE):
        try:
            with open(FEATURE_FILE, 'rb') as f:
                news_cache = pickle.load(f)
            print(f"[特征] 已加载 {len(news_cache)} 条")
            # 动态更新类别列表
            if news_cache:
                all_cats = set()
                for news in news_cache.values():
                    cat = news.get('category')
                    if cat:
                        all_cats.add(cat)
                if all_cats:
                    CATEGORIES = sorted(list(all_cats))
                    print(f"📋 动态更新类别列表: {CATEGORIES}")
            return True
        except Exception as e:
            print(f"[特征] 加载失败: {e}")
            news_cache = {}
            return False
    else:
        print(f"[特征] 文件不存在: {FEATURE_FILE}")
        news_cache = {}
        return False


# ==================== 文件监控 ====================
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class FeatureFileHandler(FileSystemEventHandler):
    """监控特征文件变化"""

    def on_modified(self, event):
        if event.src_path.endswith('juhe_news_features.pkl'):
            print("\n[特征] 检测到文件变更，重新加载中...")
            time.sleep(1)
            load_news_features()
            print(f"[特征] 已更新，共 {len(news_cache)} 条")


def start_file_monitor():
    """启动文件监控"""
    event_handler = FeatureFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=BASE_DIR, recursive=False)
    observer.start()
    print("[特征] 文件监控已启动，将自动重载特征")


# ==================== 加载模型 ====================
def load_model():
    global agent, model_loaded
    if os.path.exists(MODEL_PATH):
        try:
            test_state = np.concatenate([np.zeros(10), np.zeros(64)])
            state_dim = len(test_state)
            agent = TD3Agent(state_dim=state_dim, action_dim=1, hidden_size=256)
            agent.load(MODEL_PATH)
            model_loaded = True
            print("[模型] 加载成功")
            return True
        except Exception as e:
            print(f"[模型] 加载失败: {e}")
            return False
    else:
        print(f"[模型] 未找到文件: {MODEL_PATH}")
        return False


# 初始加载
load_news_features()
load_model()


# ==================== 工具函数 ====================
def save_feedback(feedback_data):
    try:
        feedback_data['timestamp'] = datetime.now().isoformat()
        with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')

        # 尝试写入 MySQL 反馈表（失败则忽略，继续使用文件）
        conn = get_mysql_conn()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO feedback (user_id, news_id, clicked, dwell_time, liked, reward, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            feedback_data.get('user_id'),
                            feedback_data.get('news_id'),
                            1 if feedback_data.get('clicked') else 0,
                            int(feedback_data.get('dwell_time', 0) or 0),
                            1 if feedback_data.get('liked') else 0,
                            float(feedback_data.get('reward', 0) or 0.0),
                            datetime.fromisoformat(feedback_data['timestamp']),
                        ),
                    )
            finally:
                conn.close()
        return True
    except Exception as e:
        print(f"保存反馈失败: {e}")
        return False


def save_recommendation_exposure(user_id, request_id, recommendations, exposed_at):
    """将一次推荐结果的曝光日志写入 MySQL（不可用时静默跳过）"""
    conn = get_mysql_conn()
    if conn is None:
        return False
    try:
        rows = []
        for idx, rec in enumerate(recommendations, start=1):
            rows.append(
                (
                    request_id,
                    user_id,
                    rec.get('news_id', ''),
                    idx,
                    float(rec.get('confidence', 0.0) or 0.0),
                    rec.get('category', None),
                    rec.get('source', None),
                    exposed_at,
                )
            )
        if not rows:
            return True
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO recommendation_exposure
                (request_id, user_id, news_id, rank_pos, score, category, source, exposed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
        return True
    except Exception as e:
        print(f"写入推荐曝光日志失败: {e}")
        return False
    finally:
        conn.close()


def load_users():
    conn = get_mysql_conn()
    if conn is not None:
        users = {}
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT username, interests_json, last_login, password_hash FROM users")
                for username, interests_json, last_login, password_hash in cur.fetchall():
                    try:
                        interests = interests_json if isinstance(interests_json, dict) else json.loads(interests_json)
                    except:
                        interests = {}
                    users[username] = {
                        'interests': interests,
                        'last_login': last_login.isoformat().replace(' ', 'T') if last_login else None,
                        'password_hash': password_hash or ''
                    }
            return users
        finally:
            conn.close()
    # 回退到 JSON 文件
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 确保每个用户都有 password_hash 字段（兼容旧数据）
            for user in data.values():
                if 'password_hash' not in user:
                    user['password_hash'] = ''
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_users(users):
    # 1. 先写入 JSON 文件（原子操作：先写临时文件再替换）
    try:
        temp_file = USERS_FILE + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, USERS_FILE)  # 原子替换
    except Exception as e:
        print(f"[存储] 写入 users.json 失败: {e}")
        return False  # JSON 写入失败，不继续 MySQL 操作，避免不一致

    # 2. 同步到 MySQL（幂等 upsert，但不更新 password_hash）
    conn = get_mysql_conn()
    if conn is None:
        # MySQL 不可用，仅 JSON 存储，这是允许的（可选模式）
        return True

    try:
        with conn.cursor() as cur:
            for username, u in users.items():
                interests = u.get('interests', {})
                last_login = u.get('last_login')
                password_hash = u.get('password_hash', '')  # 仅用于新用户插入
                cur.execute(
                    """
                    INSERT INTO users (username, interests_json, last_login, password_hash)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        interests_json = VALUES(interests_json),
                        last_login = VALUES(last_login)
                        -- 注意：不更新 password_hash，避免覆盖已有哈希
                    """,
                    (
                        username,
                        json.dumps(interests, ensure_ascii=False),
                        datetime.fromisoformat(last_login) if isinstance(last_login, str) else None,
                        password_hash,
                    ),
                )
        conn.commit()
        return True
    except Exception as e:
        print(f"[MySQL] 同步 users 失败: {e}")
        # JSON 已更新，MySQL 失败不影响主流程（降级模式）
        return False
    finally:
        conn.close()

def build_user_vector(interests):
    values = [float(interests.get(cat, 0.1)) for cat in CATEGORIES]
    vector_10d = np.array(values, dtype=np.float32)
    if vector_10d.sum() > 0:
        vector_10d = vector_10d / vector_10d.sum()
    return np.concatenate([vector_10d, np.zeros(54)]).tolist()


def load_user_state():
    default_state = {'users': {}}
    if not os.path.exists(USER_STATE_FILE):
        return default_state
    try:
        with open(USER_STATE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get('users'), dict):
                return data
            return default_state
    except Exception:
        return default_state


def save_user_state(state):
    with open(USER_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def ensure_user_state(state, user_id):
    users = state.setdefault('users', {})
    user_state = users.get(user_id)
    if not isinstance(user_state, dict):
        user_state = {
            'favorites': [],
            'read_history': {},
            'updated_at': datetime.now().isoformat(),
            # 时段偏好与短期兴趣（画像扩展字段）
            'hourly_category_clicks': {},   # 格式: {"0": {"科技":5, "体育":2}, ...}
            'recent_clicks': []             # 最多10条
        }
        users[user_id] = user_state
    user_state.setdefault('favorites', [])
    user_state.setdefault('read_history', {})
    user_state.setdefault('hourly_category_clicks', {})
    user_state.setdefault('recent_clicks', [])
    return user_state


def resolve_news_image(news_id, news_info):
    """
    优先使用 thumbnail_pic_s 字段，其次其他图片字段。
    """
    if not isinstance(news_info, dict):
        return ''

    # 优先取 thumbnail_pic_s（聚合数据标准字段）
    img = news_info.get('thumbnail_pic_s', '')
    if img and img.startswith(('http://', 'https://')):
        return img

    # 后备字段
    candidates = [
        news_info.get('image'),
        news_info.get('thumbnail'),
        news_info.get('thumbnail_pic_s02'),
        news_info.get('thumbnail_pic_s03'),
    ]
    for c in candidates:
        if isinstance(c, str) and c.startswith(('http://', 'https://')):
            return c


    # 2) 没有图片则尝试从新闻页抓取（轻量兜底）
    url = news_info.get('url', '')
    if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
        return ''

    try:
        from urllib.parse import urljoin
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
        }
        resp = article_session.get(url, headers=headers, timeout=(3, 6), allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # meta 优先
        meta_selectors = [
            "meta[property='og:image']",
            "meta[name='twitter:image']",
            "meta[itemprop='image']",
        ]
        image_url = ''
        for sel in meta_selectors:
            tag = soup.select_one(sel)
            if tag:
                image_url = (tag.get('content') or '').strip()
                if image_url:
                    break

        # fallback 到第一张 img
        if not image_url:
            img = soup.find('img')
            if img:
                image_url = (img.get('src') or img.get('data-src') or '').strip()

        if image_url:
            if image_url.startswith('//'):
                image_url = 'https:' + image_url
            if not image_url.startswith(('http://', 'https://')):
                image_url = urljoin(url, image_url)
            if image_url.startswith(('http://', 'https://')):
                # 回写缓存，后续请求不再抓取
                news_cache[news_id]['image'] = image_url
                return image_url
    except Exception:
        return ''

    return ''


# ==================== API 路由 ====================



@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/api/status', methods=['GET'])
def index():
    return jsonify({
        'name': '新闻推荐系统 API (TD3新模型)',
        'version': '2.0',
        'status': 'running',
        'model_loaded': model_loaded,
        'news_count': len(news_cache)
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'news_count': len(news_cache)
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        total_feedbacks = 0
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                total_feedbacks = sum(1 for _ in f)

        category_stats = {}
        for news_id, news in news_cache.items():
            cat = news.get('category', '未知')
            category_stats[cat] = category_stats.get(cat, 0) + 1
        users = load_users()

        return jsonify({
            'total_news': len(news_cache),
            'total_feedbacks': total_feedbacks,
            'total_users': len(users),
            'category_stats': category_stats,
            'model_loaded': model_loaded
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/metrics', methods=['GET'])
def admin_metrics():
    """
    管理指标接口：
    - 优先使用 MySQL 聚合
    - 若 MySQL 不可用，回退到本地 JSONL 文件
    可选参数：
    - days: 统计最近 N 天，默认 7
    """
    try:
        days = int(request.args.get('days', 7))
        if days <= 0:
            days = 7
        since_dt = datetime.now() - timedelta(days=days)

        # ---------------- MySQL 聚合 ----------------
        conn = get_mysql_conn()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM recommendation_exposure WHERE exposed_at >= %s",
                        (since_dt,),
                    )
                    exposure_count = int((cur.fetchone() or [0])[0] or 0)

                    cur.execute(
                        """
                        SELECT
                            COUNT(*) AS total_feedbacks,
                            SUM(CASE WHEN clicked=1 THEN 1 ELSE 0 END) AS clicked_count,
                            AVG(dwell_time) AS avg_dwell_time,
                            AVG(reward) AS avg_reward,
                            COUNT(DISTINCT user_id) AS feedback_users
                        FROM feedback
                        WHERE timestamp >= %s
                        """,
                        (since_dt,),
                    )
                    row = cur.fetchone() or (0, 0, 0, 0, 0)
                    total_feedbacks = int(row[0] or 0)
                    clicked_count = int(row[1] or 0)
                    avg_dwell_time = float(row[2] or 0.0)
                    avg_reward = float(row[3] or 0.0)
                    feedback_users = int(row[4] or 0)

                    cur.execute(
                        "SELECT COUNT(DISTINCT user_id) FROM recommendation_exposure WHERE exposed_at >= %s",
                        (since_dt,),
                    )
                    exposure_users = int((cur.fetchone() or [0])[0] or 0)
                    active_users = max(feedback_users, exposure_users)

                    cur.execute(
                        "SELECT COUNT(*) FROM user_favorites WHERE created_at >= %s",
                        (since_dt,),
                    )
                    favorites_added = int((cur.fetchone() or [0])[0] or 0)

                    cur.execute("SELECT COUNT(*) FROM users")
                    total_users = int((cur.fetchone() or [0])[0] or 0)

                ctr = (clicked_count / exposure_count) if exposure_count > 0 else 0.0
                favorite_rate = (favorites_added / exposure_count) if exposure_count > 0 else 0.0

                return jsonify({
                    'window_days': days,
                    'data_source': 'mysql',
                    'model_loaded': model_loaded,
                    'news_count': len(news_cache),
                    'total_users': total_users,
                    'active_users': active_users,
                    'exposure_count': exposure_count,
                    'feedback_count': total_feedbacks,
                    'clicked_count': clicked_count,
                    'ctr': round(ctr, 6),
                    'avg_dwell_time': round(avg_dwell_time, 3),
                    'favorites_added': favorites_added,
                    'favorite_rate': round(favorite_rate, 6),
                    'avg_reward': round(avg_reward, 6),
                    'generated_at': datetime.now().isoformat(),
                })
            finally:
                conn.close()

        # ---------------- 文件回退聚合 ----------------
        exposure_count = 0
        feedback_count = 0
        clicked_count = 0
        total_dwell = 0.0
        dwell_n = 0
        total_reward = 0.0
        reward_n = 0
        active_users = set()
        recommendation_users = set()

        if os.path.exists(USER_HISTORY_FILE):
            with open(USER_HISTORY_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    ts = item.get('timestamp')
                    try:
                        ts_dt = datetime.fromisoformat(ts) if ts else None
                    except Exception:
                        ts_dt = None
                    if ts_dt is not None and ts_dt < since_dt:
                        continue

                    if item.get('type') == 'recommendation':
                        exposure_count += len(item.get('recommended_news', []))
                        uid = item.get('user_id')
                        if uid:
                            recommendation_users.add(uid)
                    elif item.get('type') == 'feedback':
                        feedback_count += 1
                        uid = item.get('user_id')
                        if uid:
                            active_users.add(uid)
                        if item.get('clicked'):
                            clicked_count += 1
                        dwell = float(item.get('dwell_time', 0) or 0.0)
                        total_dwell += dwell
                        dwell_n += 1
                        reward = float(item.get('reward', 0) or 0.0)
                        total_reward += reward
                        reward_n += 1

        users = load_users()
        total_users = len(users)
        active_users_count = len(active_users.union(recommendation_users))
        avg_dwell_time = (total_dwell / dwell_n) if dwell_n > 0 else 0.0
        avg_reward = (total_reward / reward_n) if reward_n > 0 else 0.0
        ctr = (clicked_count / exposure_count) if exposure_count > 0 else 0.0

        return jsonify({
            'window_days': days,
            'data_source': 'json_files',
            'model_loaded': model_loaded,
            'news_count': len(news_cache),
            'total_users': total_users,
            'active_users': active_users_count,
            'exposure_count': exposure_count,
            'feedback_count': feedback_count,
            'clicked_count': clicked_count,
            'ctr': round(ctr, 6),
            'avg_dwell_time': round(avg_dwell_time, 3),
            'favorites_added': 0,
            'favorite_rate': 0.0,
            'avg_reward': round(avg_reward, 6),
            'generated_at': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = data.get('user_id', 'anonymous')
        user_vector = np.array(data.get('user_vector', np.zeros(64)))
        top_k = data.get('top_k', 6)
        if user_vector.shape[0] < 64:
            user_vector = np.zeros(64)

        # 如果前端未传有效向量，自动从用户画像恢复，保证可持续个性化
        if np.allclose(user_vector, 0):
            users = load_users()
            interests = users.get(user_id, {}).get('interests', {cat: 0.1 for cat in CATEGORIES})
            user_vector = np.array(build_user_vector(interests))

        if len(news_cache) == 0:
            return jsonify({'error': '没有可用新闻'}), 404

        if not model_loaded or agent is None:
            return jsonify({'error': '模型未加载'}), 503

        news_ids = list(news_cache.keys())
        random.shuffle(news_ids)

        valid_news = news_ids[:100]

        news_features = [news_cache[nid]['feature'] for nid in valid_news[:50]]
        news_mean = np.mean(news_features, axis=0)

        user_interest = user_vector[:10]
        state = np.concatenate([user_interest, news_mean])

        start_time = time.time()
        recommendations = []
        actions_list = []
        recommended_ids = []

        temp_news = valid_news.copy()
        for _ in range(min(top_k, len(temp_news))):
            action = agent.select_action(state, noise=0.3)
            action_value = float(action[0])
            actions_list.append(action_value)

            offset = np.random.uniform(-0.3, 0.3)
            modified_action = np.clip(action_value + offset, -1, 1)

            idx = int((modified_action + 1) / 2 * (len(temp_news) - 1))
            idx = max(0, min(idx, len(temp_news) - 1))
            news_id = temp_news[idx]
            recommended_ids.append(news_id)

            news_info = news_cache.get(news_id, {})
            recommendations.append({
                'news_id': news_id,
                'title': news_info.get('title', '')[:100],
                'category': news_info.get('category', ''),
                'source': news_info.get('source', ''),
                'url': news_info.get('url', ''),
                'thumbnail_pic_s': news_info.get('thumbnail_pic_s', ''),
                'image': resolve_news_image(news_id, news_info),
                'summary': news_info.get('summary', ''),
                'confidence': float(modified_action)
            })
            temp_news.pop(idx)

        response_time = time.time() - start_time

        request_id = f"{user_id}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        now_iso = datetime.now().isoformat()

        recommendation_context = {
            'type': 'recommendation',
            'request_id': request_id,
            'user_id': user_id,
            'timestamp': now_iso,
            'state': state.tolist(),
            'actions': actions_list,
            'recommended_news': recommended_ids,
            'top_k': top_k
        }

        with open(USER_HISTORY_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(recommendation_context, ensure_ascii=False) + '\n')

        state_data = load_user_state()
        user_state = ensure_user_state(state_data, user_id)

        # 短期兴趣频率
        recent_clicks = user_state.get('recent_clicks', [])
        short_freq = {}
        for cat in recent_clicks:
            short_freq[cat] = short_freq.get(cat, 0) + 1
        total_short = len(recent_clicks) if recent_clicks else 1

        # 时段偏好：当前小时各类别历史点击次数
        current_hour = str(datetime.now().hour)
        hourly_dict = user_state.get('hourly_category_clicks', {}).get(current_hour, {})

        for rec in recommendations:
            cat = rec.get('category')
            if not cat:
                continue
            short_factor = 1 + (short_freq.get(cat, 0) / total_short)
            hour_factor = 1 + math.log(1 + hourly_dict.get(cat, 0))
            rec['confidence'] *= (short_factor * hour_factor)

        # 按新置信度降序排序
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)

        # MySQL 曝光日志（用于后续 metrics 与训练分析）
        try:
            save_recommendation_exposure(
                user_id=user_id,
                request_id=request_id,
                recommendations=recommendations,
                exposed_at=datetime.fromisoformat(now_iso),
            )

        except Exception as e:
            print(f"写曝光日志异常: {e}")

        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations),
            'response_time': response_time
        })

    except Exception as e:
        print(f"推荐出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        user_id = data.get('user_id', 'anonymous')
        news_id = data.get('news_id')
        clicked = data.get('clicked', False)
        dwell_time = data.get('dwell_time', 0)
        liked = data.get('liked', False)

        if not news_id:
            return jsonify({'error': '缺少 news_id'}), 400

        reward = 0
        if clicked:
            # 根据阅读时长决定奖励/惩罚
            if dwell_time > 30:
                reward = 2.0
            elif dwell_time > 15:
                reward = 1.5
            elif dwell_time > 5:
                reward = 0.5
            elif dwell_time > 0:
                # 停留过短：视为误点或内容与预期不符
                reward = -0.3
            else:
                reward = 0.0

            if liked:
                reward += 0.5
        else:
            reward = -0.2

        # ---------- 2. 根据推荐位置调整奖励 ----------
        conn = get_mysql_conn()
        if conn:
            try:
                with conn.cursor() as cur:
                    # 查找最近一次曝光记录（before feedback timestamp）
                    cur.execute(
                        """
                        SELECT rank_pos FROM recommendation_exposure
                        WHERE user_id = %s AND news_id = %s AND exposed_at <= NOW()
                        ORDER BY exposed_at DESC LIMIT 1
                        """,
                        (user_id, news_id)
                    )
                    row = cur.fetchone()
                    if row:
                        rank_pos = row[0]
                        # 位置越靠后（数值越大），奖励倍数越高，例如：倍数 = 1 + 0.1 * log2(rank_pos+1)
                        import math
                        pos_factor = 1 + 0.1 * math.log2(rank_pos + 1)
                        reward = reward * pos_factor
            except Exception as e:
                print(f"查询曝光位置失败: {e}")
            finally:
                conn.close()

        feedback_data = {
            'type': 'feedback',
            'user_id': user_id,
            'news_id': news_id,
            'clicked': clicked,
            'dwell_time': dwell_time,
            'liked': liked,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        }

        save_feedback(feedback_data)

        with open(USER_HISTORY_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')

        # ==================== 更新用户画像（用户兴趣向量持久化） ====================
        categories = CATEGORIES
        # 默认兴趣值与 /login 保持一致
        default_interests = {cat: 0.1 for cat in categories}

        users = load_users()

        user_record = users.get(user_id, {}) if isinstance(users, dict) else {}
        interests_raw = user_record.get('interests', default_interests)
        if isinstance(interests_raw, dict):
            interests = interests_raw.copy()
        else:
            interests = default_interests.copy()

        # 根据反馈新闻的类别更新对应兴趣
        news_category = (news_cache.get(news_id, {}) or {}).get('category', '')
        if news_category in categories:
            # reward 通常在小范围内波动；用较小系数更新兴趣，避免向量崩掉
            delta = float(reward) * 0.05
            interests[news_category] = float(interests.get(news_category, 0.1)) + delta

        # 兴趣值裁剪（不允许为负，且上限避免数值爆炸）
        for cat in categories:
            interests[cat] = float(interests.get(cat, 0.1))
            interests[cat] = max(0.01, min(5.0, interests[cat]))

        # 持久化用户画像到 users.json
        users[user_id] = {
            'interests': interests,
            'last_login': datetime.now().isoformat()
        }
        save_users(users)

        news_category = (news_cache.get(news_id, {}) or {}).get('category', '')
        if news_category and clicked:
            state = load_user_state()
            user_state = ensure_user_state(state, user_id)

            # 时段偏好：当前小时
            now = datetime.now()
            hour = str(now.hour)  # "0" ~ "23"
            hourly_clicks = user_state.setdefault('hourly_category_clicks', {})
            hour_dict = hourly_clicks.setdefault(hour, {})
            hour_dict[news_category] = hour_dict.get(news_category, 0) + 1

            # 短期兴趣：维护最近10次点击
            recent = user_state.setdefault('recent_clicks', [])
            recent.append(news_category)
            max_len = 10
            if len(recent) > max_len:
                recent.pop(0)

            save_user_state(state)

        # 返回最新 user_vector（用于前端即时个性化）
        user_vector = build_user_vector(interests)

        return jsonify({
            'status': 'success',
            'reward': reward,
            'user_vector': user_vector,
            'interests': interests
        })

    except Exception as e:
        print(f"反馈出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/user/vector', methods=['POST'])
def generate_user_vector():
    try:
        data = request.json
        interests = data.get('interests', {})

        categories = CATEGORIES
        vector_64d = build_user_vector(interests)

        return jsonify({
            'user_vector': vector_64d,
            'categories': categories
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        frontend_interests = data.get('interests', {})  # 前端传来的标签权重

        if not username or not password:
            return jsonify({'error': '用户名和密码不能为空'}), 400

        # 优先从 MySQL 读取
        conn = get_mysql_conn()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT interests_json, last_login, password_hash FROM users WHERE username = %s",
                        (username,)
                    )
                    row = cur.fetchone()
                    if row:
                        interests_json, last_login, password_hash = row
                        if not password_hash:
                            return jsonify({'error': '该账号未设置密码，请联系管理员重置'}), 401
                        if not check_password_hash(password_hash, password):
                            return jsonify({'error': '密码错误'}), 401

                        # 老用户：将前端标签选择视为奖励信号，增量更新兴趣
                        if interests_json:
                            try:
                                current_interests = interests_json if isinstance(interests_json, dict) else json.loads(interests_json)
                            except:
                                current_interests = {}
                        else:
                            current_interests = {}

                        # 奖励更新：选中标签给予 +0.05 * 0.8
                        for cat, val in frontend_interests.items():
                            if val > 0.5:  # 选中标签
                                current_interests[cat] = current_interests.get(cat, 0.1) + 0.04  # 0.05 * 0.8
                        # 裁剪到 [0.01, 5.0]
                        for cat in CATEGORIES:
                            current_interests[cat] = max(0.01, min(5.0, current_interests.get(cat, 0.1)))
                        # 更新数据库中的兴趣和最后登录时间
                        cur.execute(
                            "UPDATE users SET interests_json = %s, last_login = %s WHERE username = %s",
                            (json.dumps(current_interests), datetime.now(), username)
                        )
                        conn.commit()
                        vector_64d = build_user_vector(current_interests)
                        return jsonify({
                            'status': 'success',
                            'username': username,
                            'user_vector': vector_64d,
                            'interests': current_interests
                        })
                    else:
                        # 新用户：使用前端传来的标签权重作为初始兴趣向量
                        password_hash = generate_password_hash(password)
                        # 确保所有类别都有值（缺失的补0.1）
                        new_interests = {cat: frontend_interests.get(cat, 0.1) for cat in CATEGORIES}
                        cur.execute(
                            """INSERT INTO users (username, interests_json, last_login, password_hash)
                               VALUES (%s, %s, %s, %s)""",
                            (username, json.dumps(new_interests), datetime.now(), password_hash)
                        )
                        conn.commit()
                        vector_64d = build_user_vector(new_interests)
                        return jsonify({
                            'status': 'success',
                            'username': username,
                            'user_vector': vector_64d,
                            'interests': new_interests
                        })
            finally:
                conn.close()
        else:
            # 回退到 JSON 文件
            users = load_users()
            if username in users:
                stored_hash = users[username].get('password_hash')
                if not stored_hash:
                    return jsonify({'error': '该账号未设置密码，请联系管理员重置'}), 401
                if not check_password_hash(stored_hash, password):
                    return jsonify({'error': '密码错误'}), 401

                # 老用户：增量更新兴趣
                current_interests = users[username].get('interests', {})
                for cat, val in frontend_interests.items():
                    if val > 0.5:
                        current_interests[cat] = current_interests.get(cat, 0.1) + 0.04
                for cat in CATEGORIES:
                    current_interests[cat] = max(0.01, min(5.0, current_interests.get(cat, 0.1)))
                users[username]['interests'] = current_interests
                users[username]['last_login'] = datetime.now().isoformat()
                save_users(users)
                vector_64d = build_user_vector(current_interests)
                return jsonify({
                    'status': 'success',
                    'username': username,
                    'user_vector': vector_64d,
                    'interests': current_interests
                })
            else:
                # 新用户：使用前端标签初始化
                password_hash = generate_password_hash(password)
                new_interests = {cat: frontend_interests.get(cat, 0.1) for cat in CATEGORIES}
                users[username] = {
                    'interests': new_interests,
                    'last_login': datetime.now().isoformat(),
                    'password_hash': password_hash
                }
                save_users(users)
                vector_64d = build_user_vector(new_interests)
                return jsonify({
                    'status': 'success',
                    'username': username,
                    'user_vector': vector_64d,
                    'interests': new_interests
                })

    except Exception as e:
        print(f"登录出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/user/profile', methods=['GET', 'PUT'])
def user_profile():
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
            if not user_id:
                return jsonify({'error': '缺少 user_id 参数'}), 400
            users = load_users()
            user = users.get(user_id)
            if not user:
                return jsonify({'error': '用户不存在'}), 404
            interests = user.get('interests', {})
            return jsonify({
                'user_id': user_id,
                'interests': interests,
                'user_vector': build_user_vector(interests),
                'last_login': user.get('last_login')
            })

        data = request.json or {}
        user_id = data.get('user_id')
        interests = data.get('interests', {})
        if not user_id:
            return jsonify({'error': '缺少 user_id'}), 400
        if not isinstance(interests, dict):
            return jsonify({'error': 'interests 格式错误'}), 400

        users = load_users()
        merged = {cat: float(interests.get(cat, 0.1)) for cat in CATEGORIES}
        users[user_id] = {
            'interests': merged,
            'last_login': datetime.now().isoformat()
        }
        save_users(users)
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'interests': merged,
            'user_vector': build_user_vector(merged)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/user/favorites', methods=['GET', 'POST', 'DELETE'])
def user_favorites():
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
            if not user_id:
                return jsonify({'error': '缺少 user_id 参数'}), 400
            conn = get_mysql_conn()
            if conn is not None:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT news_id FROM user_favorites WHERE user_id=%s ORDER BY created_at DESC",
                            (user_id,),
                        )
                        favorites = [row[0] for row in cur.fetchall()]
                    return jsonify({'user_id': user_id, 'favorites': favorites})
                finally:
                    conn.close()

            # 回退到文件
            state = load_user_state()
            user_state = ensure_user_state(state, user_id)
            return jsonify({'user_id': user_id, 'favorites': user_state.get('favorites', [])})

        data = request.json or {}
        user_id = data.get('user_id')
        news_id = data.get('news_id')
        if not user_id or not news_id:
            return jsonify({'error': '缺少 user_id 或 news_id'}), 400

        conn = get_mysql_conn()
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    if request.method == 'POST':
                        cur.execute(
                            """
                            INSERT IGNORE INTO user_favorites (user_id, news_id, created_at)
                            VALUES (%s, %s, %s)
                            """,
                            (user_id, news_id, datetime.now()),
                        )
                    else:
                        cur.execute(
                            "DELETE FROM user_favorites WHERE user_id=%s AND news_id=%s",
                            (user_id, news_id),
                        )
                    # 返回最新收藏列表
                    cur.execute(
                        "SELECT news_id FROM user_favorites WHERE user_id=%s ORDER BY created_at DESC",
                        (user_id,),
                    )
                    favorites = [row[0] for row in cur.fetchall()]
                return jsonify({'status': 'success', 'favorites': favorites})
            finally:
                conn.close()

        # 回退到文件
        state = load_user_state()
        user_state = ensure_user_state(state, user_id)
        favorites = user_state.get('favorites', [])

        if request.method == 'POST':
            if news_id not in favorites:
                favorites.append(news_id)
            user_state['updated_at'] = datetime.now().isoformat()
            save_user_state(state)
            return jsonify({'status': 'success', 'favorites': favorites})

        if news_id in favorites:
            favorites.remove(news_id)
        user_state['updated_at'] = datetime.now().isoformat()
        save_user_state(state)
        return jsonify({'status': 'success', 'favorites': favorites})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/user/read', methods=['GET', 'POST'])
def user_read():
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id')
            if not user_id:
                return jsonify({'error': '缺少 user_id 参数'}), 400
            conn = get_mysql_conn()
            if conn is not None:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT news_id, read_count, last_read_ts
                            FROM user_read_history
                            WHERE user_id=%s
                            """,
                            (user_id,),
                        )
                        read_history = {}
                        for news_id, read_count, last_read_ts in cur.fetchall():
                            read_history[news_id] = {
                                'readCount': int(read_count),
                                'lastRead': int(last_read_ts),
                            }
                    return jsonify({'user_id': user_id, 'read_history': read_history})
                finally:
                    conn.close()

            state = load_user_state()
            user_state = ensure_user_state(state, user_id)
            return jsonify({'user_id': user_id, 'read_history': user_state.get('read_history', {})})

        data = request.json or {}
        user_id = data.get('user_id')
        news_id = data.get('news_id')
        if not user_id or not news_id:
            return jsonify({'error': '缺少 user_id 或 news_id'}), 400

        conn = get_mysql_conn()
        now_ms = int(time.time() * 1000)
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    # upsert 读历史
                    cur.execute(
                        """
                        INSERT INTO user_read_history (user_id, news_id, read_count, last_read_ts)
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            read_count = read_count + 1,
                            last_read_ts = VALUES(last_read_ts)
                        """,
                        (user_id, news_id, 1, now_ms),
                    )
                    cur.execute(
                        "SELECT read_count, last_read_ts FROM user_read_history WHERE user_id=%s AND news_id=%s",
                        (user_id, news_id),
                    )
                    row = cur.fetchone()
                    item = {
                        'readCount': int(row[0]),
                        'lastRead': int(row[1]),
                    } if row else {'readCount': 1, 'lastRead': now_ms}
                return jsonify({'status': 'success', 'news_id': news_id, 'item': item})
            finally:
                conn.close()

        # 回退到文件
        state = load_user_state()
        user_state = ensure_user_state(state, user_id)
        read_history = user_state.get('read_history', {})
        item = read_history.get(news_id, {'readCount': 0, 'lastRead': 0})
        item['readCount'] = int(item.get('readCount', 0)) + 1
        item['lastRead'] = now_ms
        read_history[news_id] = item
        user_state['updated_at'] = datetime.now().isoformat()
        save_user_state(state)
        return jsonify({'status': 'success', 'news_id': news_id, 'item': item})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['GET'])
def search_news():
    try:
        tag = request.args.get('tag')
        limit = int(request.args.get('limit', 6))

        if not tag or tag == '':
            news_ids = list(news_cache.keys())
            random.shuffle(news_ids)
            selected = news_ids[:limit]
        else:
            filtered = []
            for nid, news in news_cache.items():
                if news.get('category') == tag:
                    filtered.append(nid)
            random.shuffle(filtered)
            selected = filtered[:limit]

            if len(selected) < limit:
                other_ids = [nid for nid in news_cache.keys() if nid not in selected]
                random.shuffle(other_ids)
                selected.extend(other_ids[:limit - len(selected)])

        results = []
        for nid in selected:
            news = news_cache[nid]
            results.append({
                'news_id': nid,
                'title': news.get('title', '')[:100],
                'category': news.get('category', ''),
                'source': news.get('source', ''),
                'url': news.get('url', ''),
                'image': resolve_news_image(nid, news),
                'summary': news.get('summary', '')
            })

        return jsonify({
            'results': results,
            'count': len(results)
        })

    except Exception as e:
        print(f"搜索出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin/refresh_news', methods=['POST'])
def refresh_news():
    try:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, 'realtime_updater.py'],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            load_news_features()
            return jsonify({
                'status': 'success',
                'message': '新闻已更新',
                'news_count': len(news_cache)
            })
        else:
            return jsonify({
                'status': 'error',
                'error': result.stderr
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({'error': '更新超时'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/article', methods=['GET'])
def get_article():
    """获取新闻正文（爬虫）"""
    news_id = request.args.get('id')
    if not news_id:
        return jsonify({'error': '缺少 id 参数'}), 400

    # 从缓存获取新闻信息
    news = news_cache.get(news_id)
    if not news:
        return jsonify({'error': '新闻不存在'}), 404

    url = news.get('url')
    if not url:
        return jsonify({'error': '新闻无URL'}), 400

    # 先查内存缓存（避免同一篇频繁抓取导致不稳定）
    now_ts = time.time()
    with article_content_cache_lock:
        cached = article_content_cache.get(news_id)
        if cached and (now_ts - cached.get('ts', 0)) < ARTICLE_CACHE_TTL_SECONDS:
            return jsonify(cached['data'])

    def normalize_text(text: str) -> str:
        # 用空格统一分隔，避免 get_text 产生的大量换行/重复空白
        return ' '.join((text or '').split())

    def extract_article(soup: BeautifulSoup, url: str):
        # 优先选择正文容器
        container = None
        for selector in ['article', 'main', '.content', '#content', '.article-content', '.post-content',
                         '.entry-content']:
            container = soup.select_one(selector)
            if container:
                break
        if not container:
            container = soup.body

        # 将容器转为字符串，再用 BeautifulSoup 解析以便操作
        from bs4 import BeautifulSoup as bs4
        container_html = str(container)
        soup2 = bs4(container_html, 'html.parser')

        # 替换所有 img 标签的 src 为代理地址
        for img in soup2.find_all('img'):
            src = img.get('src') or img.get('data-src')
            if src:
                if src.startswith('//'):
                    src = 'https:' + src
                # 生成代理 URL
                proxy_url = f"/image_proxy?url={quote(src, safe='')}"
                img['src'] = proxy_url
                # 移除可能干扰的 data-src 属性
                if 'data-src' in img.attrs:
                    del img.attrs['data-src']
            else:
                # 如果没有 src 且没有 data-src，可删除该 img 或留空
                pass

        # 获取处理后的 HTML 字符串
        content_html = str(soup2)

        # 提取所有代理图片 URL（用于返回，可选）
        images_list = [img['src'] for img in soup2.find_all('img') if
                       img.get('src') and img['src'].startswith('/image_proxy')]

        # 如果内容太少，降级为纯文本并分段
        if len(content_html) < 200:
            text = container.get_text(separator='\n\n', strip=True)
            paragraphs = text.split('\n\n')
            content_html = ''.join(f'<p>{p}</p>' for p in paragraphs if p.strip())
            images_list = []

        return content_html, images_list

    # 在 get_article 函数内
    headers = get_headers(url)

    last_error = None
    for attempt in range(3):
        try:
            # timeout=(connect, read)
            resp = article_session.get(url, headers=headers, timeout=(5, 15), allow_redirects=True)
            # 对于 403/404 等直接视为失败，走重试/兜底
            resp.raise_for_status()

            try:
                resp.encoding = resp.apparent_encoding
            except Exception:
                pass

            soup = BeautifulSoup(resp.text, 'html.parser')
            content, images = extract_article(soup, url)

            # 简单质量过滤：避免抓到空壳页面
            if content and len(content) > 50:
                data = {
                    'title': news.get('title', ''),
                    'content': content,  # 现在是 HTML 字符串
                    'images': images,  # 可保留，也可不用
                    'url': url
                }
                with article_content_cache_lock:
                    article_content_cache[news_id] = {'ts': time.time(), 'data': data}
                    # 超过容量则移除最老条目
                    if len(article_content_cache) > ARTICLE_CACHE_MAX_ITEMS:
                        oldest_key = min(
                            article_content_cache.keys(),
                            key=lambda k: article_content_cache[k].get('ts', 0)
                        )
                        article_content_cache.pop(oldest_key, None)
                return jsonify(data)

            last_error = f'正文质量不佳（长度={len(content) if content else 0}）'
        except Exception as e:
            last_error = str(e)
            print(f"爬取文章出错（第 {attempt + 1} 次）: {e}")

        # 指数退避，降低短时网络波动导致的失败概率
        time.sleep(2 ** attempt)

    return jsonify({'error': '爬取文章失败', 'detail': last_error}), 500


@app.route('/category_news', methods=['GET'])
def category_news():
    """按类别返回分页新闻列表"""
    try:
        category = request.args.get('category')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 9))

        if not category:
            return jsonify({'error': '缺少 category 参数'}), 400

        # 筛选该类别的新闻
        filtered = []
        for nid, news in news_cache.items():
            if news.get('category') == category:
                filtered.append(nid)

        # 分页
        total = len(filtered)
        total_pages = (total + page_size - 1) // page_size
        start = (page - 1) * page_size
        end = start + page_size
        page_nids = filtered[start:end]

        # 构建返回列表
        news_list = []
        for nid in page_nids:
            news = news_cache[nid]
            news_list.append({
                'news_id': nid,
                'title': news.get('title', ''),
                'category': news.get('category', ''),
                'source': news.get('source', ''),
                'publish_time': news.get('publish_time', ''),
                'url': news.get('url', ''),
                'thumbnail_pic_s': news.get('thumbnail_pic_s', ''),
                'image': resolve_news_image(nid, news),
                'summary': news.get('summary', '')
            })

        return jsonify({
            'news': news_list,
            'total': total,
            'total_pages': total_pages,
            'current_page': page,
            'page_size': page_size
        })

    except Exception as e:
        print(f"获取分类新闻出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/image_proxy')
def image_proxy():
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'missing url'}), 400
    from urllib.parse import urlparse
    parsed = urlparse(url)
    referer = f"{parsed.scheme}://{parsed.netloc}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': referer,
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    }
    try:
        resp = requests.get(url, headers=headers, timeout=(5, 10), stream=True)
        resp.raise_for_status()
        return resp.content, 200, {'Content-Type': resp.headers.get('Content-Type', 'image/jpeg')}
    except Exception as e:
        print(f"图片代理失败: {e}")
        return jsonify({'error': 'proxy failed'}), 500

# ==================== 新闻标题搜索接口 ====================
@app.route('/search_by_title', methods=['GET'])
def search_by_title():
    """根据新闻标题关键词搜索新闻"""
    keyword = request.args.get('keyword', '').strip()
    if not keyword:
        return jsonify({'error': '请提供搜索关键词'}), 400

    limit = int(request.args.get('limit', 30))
    results = []
    for nid, news in news_cache.items():
        title = news.get('title', '')
        if keyword.lower() in title.lower():
            results.append({
                'news_id': nid,
                'title': title[:100],
                'category': news.get('category', ''),
                'source': news.get('source', ''),
                'url': news.get('url', ''),
                'thumbnail_pic_s': news.get('thumbnail_pic_s', ''),
                'image': resolve_news_image(nid, news),
                'summary': news.get('summary', '')[:150]
            })
        if len(results) >= limit:
            break

    # 按标题相关度简单排序（关键词位置越靠前越相关）
    results.sort(key=lambda x: x['title'].lower().find(keyword.lower()))
    return jsonify({
        'keyword': keyword,
        'total': len(results),
        'news': results
    })



# 如果需要提供其他静态文件（如 CSS、图片），可以添加：
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("[服务] 正在启动 Flask API...")
    print("=" * 60)
    print("\n本地访问: http://127.0.0.1:5000")
    print(f"\n📋 可用接口:")
    print(f"   • POST /login        - 用户登录/注册")
    print(f"   • POST /recommend    - 获取推荐")
    print(f"   • POST /feedback     - 提交反馈")
    print(f"   • POST /user/vector  - 生成用户向量")
    print(f"   • GET  /search       - 按标签搜索新闻")
    print(f"   • GET  /stats        - 系统统计")
    print(f"   • GET  /health       - 健康检查")
    print(f"   • POST /admin/refresh_news - 手动刷新新闻")
    print(f"   • GET  /article      - 获取新闻正文（通过新闻ID）")
    print(f"   • GET  /category_news - 按类别获取分页新闻列表")
    print(f"   • GET  /admin/metrics - 管理指标聚合")
    print(f"   • GET/PUT /user/profile  - 用户画像查询/更新")
    print(f"   • GET/POST/DELETE /user/favorites - 用户收藏管理")
    print(f"   • GET/POST /user/read    - 已读历史管理")
    print("\n[服务] 已启动，Ctrl+C 停止")
    print("=" * 60)

    # 启动文件监控
    try:
        start_file_monitor()
    except Exception as e:
        print(f"[特征] watchdog 不可用，将定时重载: {e}")

    # 初始化 MySQL（如果可用）
    try:
        init_mysql()
    except Exception as e:
        print(f"[MySQL] 初始化失败: {e}")

    app.run(host='0.0.0.0', port=5000, debug=True)