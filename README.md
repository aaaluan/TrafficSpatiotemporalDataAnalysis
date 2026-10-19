## 技术栈

python3.9+、pandas、numpy、matplotlib、streamlit

## 目录结构（与课件一致）

```
traffic_project/
├── config.py        # 所有路径和阈值集中在这里
├── utils.py         # 通用工具：Haversine距离、时间、读写、日志、计时器
├── data_clean.py    # ① 数据清洗（9条规则）
├── tr_stage1.py     # ② 阶段1：速度粗筛
├── tr_stage2.py     # ③ 阶段2：手写K-Means
├── tr_stage3.py     # ④ 阶段3：规则精判
├── purpose.py       # ⑤ 出行目的识别
├── mode.py          # ⑥ 出行方式识别
├── heatmap.py       # ⑦ 实时热力图绘制
├── visualize.py     # ⑧ 可视化，生成6张分析图表
├── main.py          # 一键运行全部8个模块
├── requirements.txt # 项目依赖包
├── README.md        # 项目说明文档
├── result/          # 程序运行后自动生成
│   ├── *.csv        # 7个结果数据表
│   └── figures/     # 输出6张PNG可视化图片
└── data/            # 原始CSV数据集
```

## 环境安装

```
# 创建并激活虚拟环境
python -m venv .env-traffic
.\.env-traffic\Scripts\Activate.ps1
# 安装依赖
pip install -r requirements.txt
```



## 运行方式

1. 离线一键执行全流水线，生成结果 csv 与 6 张图表

```
python main.py
```

2. 启动 Streamlit 交互式网页（在 project_code/traffic_project 目录运行）

```
streamlit run streamlit_app.py
```

## 模块说明（全部文件，和课件一一对应）

- config.py：全局配置文件，统一存放文件路径、各类阈值参数，其它模块直接导入。
- utils.py：公共工具模块，包含 Haversine 球面距离计算、时间处理、文件读写、日志打印、计时器工具。
- data_clean.py：数据清洗模块，按照 9 条业务规则完成原始数据过滤、脏数据剔除、字段抽取。
- tr_stage1.py：轨迹处理阶段 1，速度粗筛，对用户轨迹数据进行初步过滤。
- tr_stage2.py：轨迹处理阶段 2，手写实现 K-Means 聚类算法，完成轨迹聚类。
- tr_stage3.py：轨迹处理阶段 3，基于业务规则对聚类结果精判优化。
- purpose.py：出行目的识别模块，分析用户轨迹，判断出行目的。
- mode.py：出行方式识别模块，根据轨迹特征识别出行方式。
- heatmap.py：实时热力图绘制模块，使用 matplotlib 绘制基站人流热力图，不使用 folium。
- visualize.py：可视化绘图模块，绘制二维统计图表，使用`mpl_toolkits.mplot3d.Axes3D`绘制时空 3D 分布图，共输出 6 张图。
- main.py：项目主入口，一键顺序执行全部 8 个业务模块，输出 7 份结果 csv 文件与 6 张 PNG 图片。
- requirements.txt：项目第三方依赖清单。
- README.md：项目说明文档。
- result/：输出目录，程序运行自动生成；`.csv`是 7 份结果数据表，`figures`存放 6 张可视化图片。



## 输出图表（共 6 张）

1. 基站人流热力图
2. 商圈人流时序折线图
3. 出行方式占比饼图
4. 人流量分布直方图
5. 商圈基站距离分布图
6. 时空 3D 分布图

### 📦 部署说明（接收方）
> 本压缩包**不含Python虚拟环境**，请接收方自行创建虚拟环境。
1. 解压源码
2. 在项目根目录打开终端，创建虚拟环境
```powershell
python -m venv .env-traffic
.\.env-traffic\Scripts\Activate.ps1
pip install -r requirements.txt