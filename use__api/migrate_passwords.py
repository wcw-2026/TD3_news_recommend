from werkzeug.security import generate_password_hash
import pymysql
import json

conn = pymysql.connect(
    host='127.0.0.1',
    user='root',
    password='123456',      # 修改为您的MySQL密码
    database='news_recommender'
)
cursor = conn.cursor()

# 为所有 password_hash 为 NULL 或空字符串的用户设置默认密码 '123456'
default_hash = generate_password_hash('123456')
cursor.execute("UPDATE users SET password_hash = %s WHERE password_hash IS NULL OR password_hash = ''", (default_hash,))
conn.commit()
print(f"更新了 {cursor.rowcount} 个用户的密码哈希")

cursor.close()
conn.close()
