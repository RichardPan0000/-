

# 数据库连接池

使用数据库连接池

```
class DataBase:
    def __init__(self):
        self.mysql_pool = PooledDB(creator=pymysql, maxconnections=1000, mincached=1, blocking=True, ping=1, host='localhost', port=3306, user='user', password='password', database='test_db')
        self.starrocks_pool = PooledDB(creator=pymysql, maxconnections=1000, mincached=1, blocking=True, ping=1, host='localhost', port=9030, user='user', password='password', database='test_db')

    def close(self):
        # 关闭连接池
        self.mysql_pool.close()
        self.starrocks_pool.close()

# 在应用启动时初始化连接池
db_conn = DataBase()

# 在应用退出时关闭连接池
db_conn.close()

```







```
def excute_mysql_one(self, sql):
    # 从连接池获取连接
    with self.mysql_pool.connection() as mysql_conn:
        # 使用连接执行查询
        with mysql_conn.cursor() as mysql_cursor:
            mysql_cursor.execute(sql)
            data = mysql_cursor.fetchone()
    # 返回查询结果，不需要手动关闭连接，连接会自动归还给连接池
    return data

def excute_mysql_all(self, sql):
    # 从连接池获取连接
    with self.mysql_pool.connection() as mysql_conn:
        # 使用连接执行查询
        with mysql_conn.cursor() as mysql_cursor:
            mysql_cursor.execute(sql)
            data = mysql_cursor.fetchall()
    return data

def excute_starrocks_one(self, sql):
    # 从连接池获取连接
    with self.starrocks_pool.connection() as starrocks_conn:
        # 使用连接执行查询
        with starrocks_conn.cursor() as starrocks_cursor:
            starrocks_cursor.execute(sql)
            data = starrocks_cursor.fetchone()
    return data

def excute_starrocks_all(self, sql):
    # 从连接池获取连接
    with self.starrocks_pool.connection() as starrocks_conn:
        # 使用连接执行查询
        with starrocks_conn.cursor() as starrocks_cursor:
            starrocks_cursor.execute(sql)
            data = starrocks_cursor.fetchall()
    return data

```



在使用 `with` 语句时，连接会被自动归还到连接池，而不是直接关闭。 这意味着连接池会管理连接的生命周期，确保连接在使用后可以被其他请求复用。 因此，您的代码是正确的，连接会在 `with` 语句块结束时自动归还给连接池，无需手动关闭。



https://webwareforpython.github.io/DBUtils/main.html#pooleddb-pooled-db

![image-20250211150351463](C:\Users\100488\AppData\Roaming\Typora\typora-user-images\image-20250211150351463.png)