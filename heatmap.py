# heatmap.py
"""
 实时商圈热力分析模块
输入：data/实时流数据.csv、data/商业区数据.csv
输出：result/heatmap_result.csv （时间-商圈-人次）
使用预计算+rolling滑动累加优化性能
"""
import os
import pandas as pd
import numpy as np

from config import (
    DATA_DIR, RESULT_DIR, FLOW_FILE, AREA_FILE,
    HEATMAP_MATCH_THRESHOLD, HEATMAP_WINDOW_SIZE, BATCH_SIZE, TARGET_DATE,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
)
from utils import haversine


def heatmap_analysis():
    """完整实时商圈热力分析：读取实时流→预处理→商圈空间匹配→滑动窗口统计→输出heatmap_result.csv"""
    print("=" * 60)
    print("【实时商圈热力分析 heatmap.py】")
    print("=" * 60)

    # --------步骤1 读取实时流数据(无表头)--------
    stream_path = os.path.join(DATA_DIR, FLOW_FILE)
    stream_df = pd.read_csv(stream_path, header=None, dtype=str)
    stream_df.columns = ["time", "基站编码", "lon", "lat"]
    print(f"步骤1｜读取实时流原始数据：{len(stream_df)} 行")

    # --------步骤2 预处理、时间解析、目标日期过滤--------
    stream_df["time_dt"] = pd.to_datetime(stream_df["time"], format="%Y%m%d%H%M%S", errors="coerce")
    target_dt = pd.to_datetime(TARGET_DATE, format="%Y%m%d").date()
    stream_df = stream_df[stream_df["time_dt"].dt.date == target_dt].copy()

    # 数值转换+经纬度范围过滤
    stream_df["lon"] = pd.to_numeric(stream_df["lon"], errors="coerce")
    stream_df["lat"] = pd.to_numeric(stream_df["lat"], errors="coerce")
    stream_df = stream_df.dropna(subset=["time_dt", "lon", "lat"])
    stream_df = stream_df[(stream_df["lon"].between(LON_MIN, LON_MAX)) &
                          (stream_df["lat"].between(LAT_MIN, LAT_MAX))]
    stream_df = stream_df.sort_values("time_dt").reset_index(drop=True)
    print(f"步骤2｜范围过滤完成，有效实时流：{len(stream_df)} 行")

    # --------步骤3 读取商业区数据(无表头)--------
    area_path = os.path.join(DATA_DIR, AREA_FILE)
    biz_df = pd.read_csv(area_path, header=None, dtype=str)
    biz_df.columns = ["lon", "lat", "biz_name"]
    biz_df["lon"] = pd.to_numeric(biz_df["lon"], errors="coerce")
    biz_df["lat"] = pd.to_numeric(biz_df["lat"], errors="coerce")
    biz_df = biz_df.dropna(subset=["lon", "lat"])
    biz_lons = biz_df["lon"].values
    biz_lats = biz_df["lat"].values
    biz_names = biz_df["biz_name"].values
    print(f"步骤3｜读取商业区，有效商圈数量：{len(biz_df)}")

    # --------步骤4 向量化批量空间匹配--------
    matched_biz = []
    matched_dist = []
    total = len(stream_df)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = stream_df.iloc[start:end]
        lons = batch["lon"].values[:, np.newaxis]
        lats = batch["lat"].values[:, np.newaxis]
        dists = haversine(lons, lats, biz_lons[np.newaxis, :], biz_lats[np.newaxis, :], unit="m")
        min_idx = np.argmin(dists, axis=1)
        min_dist = dists[np.arange(len(batch)), min_idx]
        batch_biz = np.where(min_dist < HEATMAP_MATCH_THRESHOLD, biz_names[min_idx], "未命中")
        matched_biz.extend(batch_biz.tolist())
        matched_dist.extend(min_dist.tolist())
        if end % 50000 == 0 or end == total:
            print(f"  空间匹配进度 {end}/{total}")

    stream_df["biz_name"] = matched_biz
    stream_df["biz_dist_m"] = matched_dist
    match_rate = (stream_df["biz_name"] != "未命中").mean() * 100
    print(f"步骤4｜商圈空间匹配完成，匹配成功率 {match_rate:.1f}%")

    # --------步骤5 预计算分钟统计 + rolling滑动窗口累加（Day09）--------
    stream_df["minute"] = stream_df["time_dt"].dt.floor("1min")
    stream_valid = stream_df[stream_df["biz_name"] != "未命中"].copy()
    minute_stats = stream_valid.groupby(["minute", "biz_name"]).size().unstack(fill_value=0)

    time_start = stream_df["time_dt"].min().floor("1min")
    time_end = stream_df["time_dt"].max().ceil("1min")
    full_minutes = pd.date_range(time_start, time_end, freq="1min")
    minute_stats = minute_stats.reindex(full_minutes, fill_value=0)

    window_stats = minute_stats.rolling(window=HEATMAP_WINDOW_SIZE, min_periods=1).sum()
    heatmap_df = window_stats.stack().reset_index()
    heatmap_df.columns = ["time", "biz_name", "count"]
    heatmap_df["time"] = heatmap_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # --------步骤6 输出结果文件--------
    out_csv = os.path.join(RESULT_DIR, "heatmap_result.csv")
    heatmap_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"步骤6｜输出热力结果 {out_csv}，总行数 {len(heatmap_df)}")
    print(f"总窗口统计人次：{heatmap_df['count'].sum():.0f}")
    print("===== 实时商圈热力模块完成 =====\n")
    return heatmap_df


if __name__ == "__main__":
    heatmap_analysis()
