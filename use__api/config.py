# config.py
# API配置
class APIConfig:
    # 聚合数据 - 新闻头条API
    JUHE_NEWS_API = {
        'key': '966ccaca61dab587478eac40f17c2bd5',  # 生产环境请改为环境变量或本地私有配置
        'base_url': 'http://v.juhe.cn/toutiao/index',
        'types': {
            'top': '头条',
            'shehui': '社会',
            'guonei': '国内',
            'guoji': '国际',
            'yule': '娱乐',
            'tiyu': '体育',
            'junshi': '军事',
            'keji': '科技',
            'caijing': '财经',
            'shishang': '时尚'
        }
    }

    # 新闻缓存设置
    CACHE_SETTINGS = {
        'expire_time': 300,  # 缓存过期时间（秒）
        'max_news': 500  # 最多缓存的新闻数量
    }