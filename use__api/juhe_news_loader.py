# juhe_news_loader.py
import requests
import time
import pickle
import os
import numpy as np
from datetime import datetime
from functools import lru_cache
from config import APIConfig


class JuheNewsLoader:
    """聚合数据 - 新闻头条API加载器"""

    def __init__(self, api_key=None, cache_dir='news_cache'):
        """
        初始化聚合数据新闻加载器

        Args:
            api_key: 聚合数据API Key，如果不提供则从配置读取
            cache_dir: 缓存目录
        """
        self.api_key = api_key or APIConfig.JUHE_NEWS_API['key']
        self.base_url = APIConfig.JUHE_NEWS_API['base_url']
        self.news_types = APIConfig.JUHE_NEWS_API['types']
        self.cache_dir = cache_dir
        self.cache_expire = APIConfig.CACHE_SETTINGS['expire_time']

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 新闻数据缓存
        self.news_cache = {}
        self.last_fetch_time = {}

        print("[Juhe] 新闻加载器已初始化")
        print(f"   API Key: {self.api_key[:5]}...{self.api_key[-5:]}")
        print(f"   支持类别: {len(self.news_types)} 种")
        print(f"   缓存目录: {cache_dir}")

    def fetch_news_by_type(self, news_type='top', page=1, page_size=30, use_cache=True):
        """
        按类别获取新闻

        Args:
            news_type: 新闻类型 (top, shehui, guonei, keji 等)
            page: 页码
            page_size: 每页数量 (最大30)
            use_cache: 是否使用缓存

        Returns:
            新闻列表
        """
        cache_key = f"{news_type}_{page}_{page_size}"

        # 检查缓存
        if use_cache and cache_key in self.last_fetch_time:
            time_diff = time.time() - self.last_fetch_time[cache_key]
            if time_diff < self.cache_expire and cache_key in self.news_cache:
                print(f"   [缓存] 使用 {news_type} 类别缓存")
                return self.news_cache[cache_key]

        # 构建请求参数
        params = {
            'key': self.api_key,
            'type': news_type,
            'page': page,
            'page_size': page_size
        }

        try:
            print(f"   [API] 获取 {self.news_types.get(news_type, news_type)} 新闻...")
            response = requests.get(self.base_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data['error_code'] == 0:
                    news_list = data['result']['data']

                    # 处理新闻数据
                    processed_news = self._process_news_list(news_list, news_type)

                    # 更新缓存
                    self.news_cache[cache_key] = processed_news
                    self.last_fetch_time[cache_key] = time.time()

                    print(f"     获取到 {len(processed_news)} 条新闻")
                    return processed_news
                else:
                    print(f"   [Juhe] API 错误: {data['reason']}")
                    return []
            else:
                print(f"   [Juhe] HTTP 错误: {response.status_code}")
                return []

        except requests.exceptions.Timeout:
            print("   [Juhe] 请求超时")
            return []
        except Exception as e:
            print(f"   [Juhe] 请求异常: {e}")
            return []

    def _process_news_list(self, news_list, news_type):
        """处理新闻列表，提取有用字段"""
        processed = []

        for news in news_list:
            # 生成唯一新闻ID
            news_id = f"juhe_{news.get('uniquekey', hash(news['title']))}"

            # 处理发布时间
            publish_time = news.get('date', datetime.now().strftime('%Y-%m-%d %H:%M'))

            # 提取有用字段
            processed_news = {
                'news_id': news_id,
                'title': news.get('title', ''),
                'category': self.news_types.get(news_type, '综合'),
                'category_code': news_type,
                'publish_time': publish_time,
                'author': news.get('author_name', ''),
                'source': news.get('author_name', ''),
                'content': news.get('content', ''),
                'url': news.get('url', ''),
                'thumbnail': news.get('thumbnail_pic_s', ''),
                # 用于特征提取的文本组合
                'text_for_features': f"{news.get('title', '')} {news.get('content', '')[:200]}"
            }

            processed.append(processed_news)

        return processed

    def fetch_multiple_categories(self, categories=None, max_per_category=20):
        """
        获取多个类别的新闻

        Args:
            categories: 类别列表，None表示获取所有类别
            max_per_category: 每类最多获取条数

        Returns:
            合并后的新闻列表
        """
        if categories is None:
            categories = list(self.news_types.keys())

        all_news = []

        print("\n[Juhe] 拉取多类别新闻...")
        for category in categories:
            news_list = self.fetch_news_by_type(
                news_type=category,
                page_size=min(max_per_category, 30),
                use_cache=True
            )
            all_news.extend(news_list)
            time.sleep(1)  # 避免请求过快

        # 去重
        unique_news = {}
        for news in all_news:
            unique_news[news['news_id']] = news

        unique_list = list(unique_news.values())
        print(f"\n[Juhe] 去重后共 {len(unique_list)} 条")

        return unique_list

    def generate_features(self, news_list, feature_dim=64, save_path='juhe_news_features.pkl'):
        """
        为新闻生成特征向量

        Args:
            news_list: 新闻列表
            feature_dim: 特征维度
            save_path: 保存路径
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        import jieba

        if not news_list:
            print("[Juhe] 新闻列表为空，跳过特征生成")
            return {}

        print("\n[Juhe] 生成新闻特征向量...")

        # 提取文本
        texts = []
        news_ids = []

        for news in news_list:
            text = news.get('text_for_features', '')
            if not text:
                text = news['title']
            texts.append(text)
            news_ids.append(news['news_id'])

        # 中文分词器
        def chinese_tokenizer(text):
            return jieba.lcut(text)

        # 使用TF-IDF生成特征
        vectorizer = TfidfVectorizer(
            max_features=feature_dim,
            tokenizer=chinese_tokenizer,
            token_pattern=None
        )

        features = vectorizer.fit_transform(texts).toarray()

        # 归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        features = features / norms

        # 构建特征字典
        news_features = {}
        for i, news_id in enumerate(news_ids):
            news_features[news_id] = {
                'feature': features[i],
                'title': news_list[i]['title'][:100],
                'category': news_list[i]['category'],
                'source': news_list[i]['source'],
                'publish_time': news_list[i]['publish_time'],
                'url': news_list[i]['url'],
                'thumbnail_pic_s': news_list[i].get('thumbnail', ''),
            }

        # 保存特征
        with open(save_path, 'wb') as f:
            pickle.dump(news_features, f)

        print(f"[Juhe] 特征生成完成: {len(news_features)} 条, 维度 {feature_dim}")
        print(f"[Juhe] 已保存: {save_path}")

        return news_features

    def load_features(self, filepath='juhe_news_features.pkl'):
        """加载已保存的特征"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                news_features = pickle.load(f)
            print(f"[Juhe] 已加载特征 {len(news_features)} 条")
            return news_features
        else:
            print(f"[Juhe] 特征文件不存在: {filepath}")
            return {}