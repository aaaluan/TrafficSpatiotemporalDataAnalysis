import os
# 数据路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

# 数据文件
RAW_FILE = "原始数据.csv"
BS_FILE = "基站经纬度数据.csv"
AREA_FILE = "商业区数据.csv"
PURPOSE_FILE = "出行目的静态数据.csv"
MODE_FILE = "出行方式静态数据.csv"
FLOW_FILE = "实时流数据.csv"
HIVE_FILE = "hive模块静态数据.csv"

# 原始数据10列（无表头）
RAW_COLS = ["timestamp", "imsi", "lac_id", "cell_id", "phone",
            "timestamp1", "tmp0", "tmp1", "nid", "npid"]

# 分析参数
TARGET_DATE = "20181003"  # 目标日期
SAMPLE_ROWS = None       # 调试用抽样20000；最终跑全量再改为None
MIN_RECORDS_PER_USER = 20   # 单用户最少记录数

# 沈阳经纬度范围
LON_MIN, LON_MAX = 122.0, 124.0
LAT_MIN, LAT_MAX = 41.0, 42.5

# 阶段1参数
LOW_FREQ_THRESHOLD = 4         # 低频点阈值
TRAVEL_SPEED_THRESHOLD = 8.0   # km/h
ABNORMAL_SPEED_THRESHOLD = 200 # km/h

# 阶段2参数（k-means聚类）
K_MIN = 2                   # 最小簇数
K_MAX = 10                  # 最大簇数
RANDOM_STATE = 42           # 随机种子
KMEANS_MAX_ITER = 100       # k-means 最大迭代次数
KMEANS_TOL = 1e-4           # k-means 收敛阈值

# 阶段3参数（规则精判）
DIST_THRESHOLD = 1000       # 距簇中心距离阈值（米）
TIME_THRESHOLD = 2          # 与下点时间跨度阈值（小时）

# purpose参数（判断出行目的）
LIVING_DURATION_THRESHOLD = 4  # 是否为居住条件的临界点
PURPOSE_MATCH_THRESHOLD = 1000  # 空间匹配的阈值

# mode 参数（出行方式）
MATCH_THRESHOLD = 500  # 空间匹配阈值
SUBWAY_MIN_POINTS = 2  # 地铁最小匹配点数
BUS_MIN_POINTS = 3     # 公交门槛3个点（公交站点密集）
SPEED_OTHER = 12  # 其他速度上限
SPEED_BUS = 20  # 公交速度上限

# =========heatmap商圈热力参数 =========
HEATMAP_MATCH_THRESHOLD = 1000   # 商圈匹配半径 米
HEATMAP_WINDOW_SIZE = 10         # 滑动窗口大小，单位分钟
BATCH_SIZE = 10000               # 分批广播批大小
