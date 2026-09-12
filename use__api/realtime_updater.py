# realtime_updater.py
import schedule
import time
import pickle
import os
import numpy as np
from datetime import datetime
import threading
import signal
import sys
from juhe_news_loader import JuheNewsLoader

class RealtimeUpdater:
    """
    实时新闻更新服务
    自动定期获取最新新闻，更新推荐系统的新闻库
    """

    def __init__(self, api_key, update_interval=3600, max_news=2000):
        """
        初始化实时更新服务

        Args:
            api_key: 聚合数据API Key
            update_interval: 更新间隔（秒），默认1小时
            max_news: 最大新闻缓存数量
        """
        self.api_key = api_key
        self.update_interval = update_interval
        self.max_news = max_news
        self.loader = JuheNewsLoader(api_key=api_key)

        # 获取脚本所在目录的绝对路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # 特征文件路径（强制保存在脚本目录下）
        self.feature_file = os.path.join(self.base_dir, 'juhe_news_features.pkl')
        self.backup_file = os.path.join(self.base_dir, 'juhe_news_features_backup.pkl')

        # 当前新闻特征
        self.news_features = self._load_existing_features()

        # 运行状态
        self.is_running = False
        self.last_update_time = None
        self.update_count = 0

        print("=" * 60)
        print("实时新闻特征更新服务")
        print("=" * 60)
        print(f"API Key: {api_key[:5]}...{api_key[-5:]}")
        print(f"更新间隔: {update_interval} 秒（约 {update_interval / 3600:.1f} 小时）")
        print(f"最大新闻条数: {max_news}")
        print(f"特征文件: {self.feature_file}")
        print(f"备份文件: {self.backup_file}")
        print(f"当前已加载新闻: {len(self.news_features)} 条")

    def _load_existing_features(self):
        """加载已有的特征文件"""
        if os.path.exists(self.feature_file):
            try:
                with open(self.feature_file, 'rb') as f:
                    features = pickle.load(f)
                print(f"[特征] 已加载本地缓存 {len(features)} 条")
                return features
            except Exception as e:
                print(f"[特征] 加载失败: {e}")
                return {}
        return {}

    def _save_features(self, features, is_backup=False):
        """保存特征到文件"""
        try:
            filepath = self.backup_file if is_backup else self.feature_file
            with open(filepath, 'wb') as f:
                pickle.dump(features, f)
            if not is_backup:
                print(f"[特征] 已保存: {filepath}")
            return True
        except Exception as e:
            print(f"[特征] 保存失败: {e}")
            return False

    def _merge_features(self, new_features):
        # 合并新旧特征
        merged = {**self.news_features, **new_features}
        if len(merged) > self.max_news:
            # 按 added_time 降序排序（最新的在前）
            items = sorted(merged.items(),
                           key=lambda x: x[1].get('added_time', ''),
                           reverse=True)
            # 只保留前 max_news 条
            merged = dict(items[:self.max_news])
        return merged

    def update_news(self):
        """执行一次新闻更新"""
        start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"[更新] 第 {self.update_count + 1} 次开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        try:
            # 1. 获取多类别新闻
            categories = list(self.loader.news_types.keys())  # 自动使用所有10类
            print(f"[更新] 拉取 {len(categories)} 个类别...")

            news_list = self.loader.fetch_multiple_categories(
                categories=categories,
                max_per_category=20  # 每类获取20条
            )

            if not news_list:
                print("[更新] 未获取到新新闻")
                return False

            print(f"[更新] 拉取到 {len(news_list)} 条新闻")

            # 2. 为新新闻生成特征
            print("[更新] 生成 TF-IDF 特征...")
            new_features = {}

            from sklearn.feature_extraction.text import TfidfVectorizer
            import jieba

            # 提取文本
            texts = []
            news_ids = []
            for news in news_list:
                text = news.get('text_for_features', news['title'])
                texts.append(text)
                news_ids.append(news['news_id'])

            # 中文分词
            def chinese_tokenizer(text):
                return jieba.lcut(text)

            # 生成特征
            vectorizer = TfidfVectorizer(
                max_features=64,
                tokenizer=chinese_tokenizer,
                token_pattern=None
            )

            features = vectorizer.fit_transform(texts).toarray()

            # 归一化
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            features = features / norms

            # 构建新特征字典
            for i, news_id in enumerate(news_ids):
                # 优先使用 thumbnail 字段（来自 juhe_news_loader）
                image = news_list[i].get('thumbnail', '')   # 直接获取 thumbnail 字段
                new_features[news_id] = {
                    'feature': features[i],
                    'title': news_list[i]['title'][:100],
                    'category': news_list[i]['category'],
                    'source': news_list[i]['source'],
                    'publish_time': news_list[i]['publish_time'],
                    'url': news_list[i]['url'],
                    'thumbnail_pic_s': image,  # 统一使用 thumbnail_pic_s
                    'summary': news_list[i].get('summary', ''),
                    'added_time': datetime.now().isoformat()
                }

            # 3. 合并新旧特征
            old_count = len(self.news_features)
            self.news_features = self._merge_features(new_features)
            new_count = len(self.news_features)

            print(f"[更新] 统计:")
            print(f"   - 原有新闻: {old_count} 条")
            print(f"   - 新增新闻: {len(new_features)} 条")
            print(f"   - 现有新闻: {new_count} 条")

            # 4. 保存到文件
            self._save_features(self.news_features)

            # 5. 创建备份
            if self.update_count % 5 == 0:  # 每5次更新备份一次
                self._save_features(self.news_features, is_backup=True)
                print("[更新] 已写入周期备份文件")

            # 更新状态
            self.last_update_time = datetime.now()
            self.update_count += 1

            print(f"[更新] 完成，耗时 {time.time() - start_time:.1f} 秒")
            return True

        except Exception as e:
            print(f"[更新] 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_job(self):
        """定时执行的更新任务"""
        print("\n[调度] 触发定时更新")
        self.update_news()

    def start(self):
        """启动实时更新服务"""
        self.is_running = True

        print("\n[服务] 实时更新已启动")
        print("首次更新约在 10 秒后执行")
        print(f"周期: 每 {self.update_interval / 3600:.1f} 小时")
        print("Ctrl+C 停止\n")

        # 10秒后执行第一次更新
        threading.Timer(10, self.update_job).start()

        # 设置定时任务
        schedule.every(self.update_interval).seconds.do(self.update_job)

        # 运行调度器
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)

    def stop(self):
        """停止实时更新服务"""
        self.is_running = False
        print("\n[服务] 实时更新已停止")

        # 保存最终状态
        self._save_features(self.news_features)
        print(f"[特征] 当前共 {len(self.news_features)} 条")


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    print("\n\n[信号] 收到中断，正在退出...")
    if 'updater' in globals():
        updater.stop()
    sys.exit(0)


def main():
    """主函数"""
    # 配置
    API_KEY = ""  # 替换成你的API Key

    if API_KEY == "966ccaca61dab587478eac40f17c2bd5":
        print("[配置] 请在 realtime_updater.py 的 main() 中填写有效的聚合数据 API Key")
        return

    # 创建更新服务实例
    global updater
    updater = RealtimeUpdater(
        api_key=API_KEY,
        update_interval=3600,  # 1小时
        max_news=2000  # 最多保存2000条新闻
    )

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)

    # 启动服务
    updater.start()


if __name__ == "__main__":
    main()