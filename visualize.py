# visualize.py
"""
Day10 综合可视化模块，共6张标准图表
1. fig_stage3_pie 出行/驻留标签分布饼图
2. fig_purpose_bar 出行目的分布柱状图
3. fig_mode_bar 出行方式分布柱状图
4. fig_hour_line 24小时出行人次折线图
5. fig_heatmap  Top10商圈 × 24小时人次热力图（读取heatmap_result.csv）
6. fig_3d_traj  Top5用户3D轨迹图（mpl_toolkits.mplot3d）

每个绘图函数返回fig对象，可离线main保存png，也可streamlit网页 st.pyplot()调用
统一配色取自课件，调用utils.setup_chinese_font()设置中文
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import RESULT_DIR, FIG_DIR
from utils import setup_chinese_font, parse_time_to_dt

setup_chinese_font()

# =========课件统一配色========
COLOR_DICT = {
    "驻留": "#2ca02c",
    "出行": "#d62728",
    "居住": "#1f77b4",
    "购物": "#ff7f0e",
    "旅游": "#9467bd",
    "其他": "#7f7f7f",
    "地铁": "#d62728",
    "公交": "#1f77b4",
    "小汽车": "#ff7f0e"
}


def plot_stage3_pie(df_label=None):
    """图1：出行/驻留标签分布饼图"""
    if df_label is None:
        df_label = pd.read_csv(os.path.join(RESULT_DIR, "stage3_label.csv"), dtype={"imsi": str})
    cnt = df_label["label"].value_counts()
    labels = ["驻留" if x == 0 else "出行" for x in cnt.index]
    colors = [COLOR_DICT["驻留"], COLOR_DICT["出行"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(cnt.values, labels=labels, autopct="%1.1f%%",
                                      colors=colors, startangle=90, textprops={"fontsize": 12})
    for autot in autotexts:
        autot.set_color("white")
        autot.set_fontweight("bold")
    ax.set_title("出行/驻留标签分布", fontsize=14, )
    plt.tight_layout()
    return fig


def plot_purpose_bar(df_purpose=None):
    """图2：出行目的分布柱状图"""
    if df_purpose is None:
        df_purpose = pd.read_csv(os.path.join(RESULT_DIR, "purpose_result.csv"))
    cnt = df_purpose["final_purpose"].value_counts()
    order = ["居住", "购物", "旅游", "其他"]
    cnt = cnt.reindex([i for i in order if i in cnt.index])
    bar_colors = [COLOR_DICT[k] for k in cnt.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cnt.index, cnt.values, color=bar_colors, edgecolor="black")
    ax.set_title("出行目的分布", fontsize=14, )
    ax.set_ylabel("驻留段数")
    for bar, val in zip(bars, cnt.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5, f"{val}\n({val / len(df_purpose) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig


def plot_mode_bar(df_mode=None):
    """图3：出行方式分布柱状图"""
    if df_mode is None:
        df_mode = pd.read_csv(os.path.join(RESULT_DIR, "mode_result.csv"))
    cnt = df_mode["final_mode"].value_counts()
    order = ["地铁", "公交", "小汽车", "其他"]
    cnt = cnt.reindex([i for i in order if i in cnt.index])
    bar_colors = [COLOR_DICT[k] for k in cnt.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cnt.index, cnt.values, color=bar_colors, edgecolor="black")
    ax.set_title("出行方式分布", fontsize=14, )
    ax.set_ylabel("出行段数")
    for bar, val in zip(bars, cnt.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val}\n({val / len(df_mode) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig


def plot_24h_travel_line(df_label=None):
    """图4：24小时出行人次折线图（只统计label=1出行点）"""
    if df_label is None:
        df_label = pd.read_csv(os.path.join(RESULT_DIR, "stage3_label.csv"), dtype={"imsi": str})
    df_label["time_dt"] = parse_time_to_dt(df_label["time"])
    df_travel = df_label[df_label["label"] == 1]
    df_travel["hour"] = df_travel["time_dt"].dt.hour
    hour_cnt = df_travel.groupby("hour").size().reindex(range(0, 24), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hour_cnt.index, hour_cnt.values, marker="o", color=COLOR_DICT["出行"], linewidth=2)
    ax.fill_between(hour_cnt.index, hour_cnt.values, alpha=0.2, color=COLOR_DICT["出行"])
    ax.set_title("24小时出行人次折线图", fontsize=14, )
    ax.set_xlabel("小时")
    ax.set_ylabel("出行人次")
    ax.set_xticks(range(0, 24))
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig


def plot_biz_hour_heatmap(df_heat=None):
    """图5：Top10商圈 × 24小时人次热力图【读取heatmap_result.csv】"""
    if df_heat is None:
        df_heat = pd.read_csv(os.path.join(RESULT_DIR, "heatmap_result.csv"))
    df_heat["time_dt"] = pd.to_datetime(df_heat["time"])
    biz_total = df_heat.groupby("biz_name")["count"].sum().sort_values(ascending=False)
    top10 = biz_total.head(10).index.tolist()
    mat = np.zeros((len(top10), 24))
    for i, biz in enumerate(top10):
        sub = df_heat[df_heat["biz_name"] == biz]
        for h in range(24):
            mat[i, h] = sub[sub["time_dt"].dt.hour == h]["count"].sum()
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(24))
    ax.set_xticklabels(list(range(24)))
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10)
    ax.set_title("Top10商圈每小时人次热力图", fontsize=14, )
    ax.set_xlabel("小时")
    ax.set_ylabel("商圈")
    plt.colorbar(im, ax=ax, label="人次")
    plt.tight_layout()
    return fig


def plot_top5_3d_traj(df_label=None):
    """
    图6：Top5用户3D移动轨迹图
    使用mpl_toolkits.mplot3d.Axes3D
    X：经度，Y：纬度，Z：时间(小时+分钟)
    """
    if df_label is None:
        df_label = pd.read_csv(os.path.join(RESULT_DIR, "stage3_label.csv"), dtype={"imsi": str})
    df_label["time_dt"] = parse_time_to_dt(df_label["time"])
    df_label = df_label.dropna(subset=["time_dt", "lon", "lat"])
    # 取点数最多Top5用户
    top5_users = df_label.groupby("imsi").size().sort_values(ascending=False).head(5).index.tolist()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    color_list = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    for idx, user in enumerate(top5_users):
        ud = df_label[df_label["imsi"] == user].sort_values("time_dt")
        x = ud["lon"].values
        y = ud["lat"].values
        z = ud["time_dt"].dt.hour.values + ud["time_dt"].dt.minute.values / 60.0
        ax.plot(x, y, z, color=color_list[idx], linewidth=1.4, label=f"用户{idx+1}")

    ax.set_xlabel("经度", labelpad=8)
    ax.set_ylabel("纬度", labelpad=8)
    ax.set_zlabel("时间（小时）", labelpad=8)
    ax.set_title("Top5用户3D移动轨迹", fontsize=14, )
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=50)
    plt.tight_layout()
    return fig


def visualize_all_save(out_dir=FIG_DIR):
    """离线批量全部6张图保存png，main.py调用"""
    os.makedirs(out_dir, exist_ok=True)
    print("\n===== visualize.py 开始生成6张标准可视化图 =====")
    f1 = plot_stage3_pie()
    f1.savefig(os.path.join(out_dir, "fig_stage3_pie.png"), dpi=150, bbox_inches="tight"); plt.close(f1)

    f2 = plot_purpose_bar()
    f2.savefig(os.path.join(out_dir, "fig_purpose_bar.png"), dpi=150, bbox_inches="tight"); plt.close(f2)

    f3 = plot_mode_bar()
    f3.savefig(os.path.join(out_dir, "fig_mode_bar.png"), dpi=150, bbox_inches="tight"); plt.close(f3)

    f4 = plot_24h_travel_line()
    f4.savefig(os.path.join(out_dir, "fig_hour_line.png"), dpi=150, bbox_inches="tight"); plt.close(f4)

    f5 = plot_biz_hour_heatmap()
    f5.savefig(os.path.join(out_dir, "fig_heatmap.png"), dpi=150, bbox_inches="tight"); plt.close(f5)

    f6 = plot_top5_3d_traj()
    f6.savefig(os.path.join(out_dir, "fig_3d_traj.png"), dpi=150, bbox_inches="tight"); plt.close(f6)
    print(f"全部6张图片输出到 {out_dir}")


if __name__ == "__main__":
    visualize_all_save()
