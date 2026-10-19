import numpy as np
import time

def haversine(lon1, lat1, lon2, lat2, unit="km"):
    """计算两点间的球面距离（向量化）。

    """
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    d = R * c
    return d * 1000 if unit == "m" else d

def setup_chinese_font():
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei',"Microsoft YaHei","Arial Unicode MS"]
    plt.rcParams['axes.unicode_minus'] = False

def parse_time_to_dt(time_series):
    import pandas as pd
    return pd.to_datetime(time_series,format="%Y%m%d%H%M%S",errors="coerce")


# ============================================================
# 计时器（上下文管理器）
# ============================================================
class Timer:
    """用 with 语句统计一段代码耗时：

    with Timer("数据清洗"):
        ...
    """

    def __init__(self, name="任务"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        print(f"  [计时] {self.name} 耗时 {elapsed:.2f} 秒")
        return False

# 加到utils.py
def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def log_step(idx, name):
    """打印模块步骤标题"""
    print(f"\n>>>>> 第{idx}步：{name} <<<<<")

def read_csv_gbk(file_name, data_dir=None):
    """
    读取csv，优先gbk编码，失败自动切换utf-8
    :param file_name: 文件名
    :param data_dir: 读取目录，不传默认DATA_DIR
    :return: DataFrame
    """
    import pandas as pd
    import os
    from config import DATA_DIR
    if data_dir is None:
        fp = os.path.join(DATA_DIR, file_name)
    else:
        fp = os.path.join(data_dir, file_name)
    try:
        df = pd.read_csv(fp, encoding="gbk")
    except UnicodeDecodeError:
        df = pd.read_csv(fp, encoding="utf-8-sig")
    return df


def save_csv(df, filepath):
    """
    保存csv，统一utf-8-sig编码，防止中文乱码
    """
    import pandas as pd
    df.to_csv(filepath, index=False, encoding="utf-8-sig")

