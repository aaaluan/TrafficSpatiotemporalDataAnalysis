import os
import pandas as pd
import numpy as np
from config import (RESULT_DIR, LOW_FREQ_THRESHOLD, TRAVEL_SPEED_THRESHOLD,
                    ABNORMAL_SPEED_THRESHOLD)
from utils import haversine


def stage1():
    print("=" * 60)
    print("阶段1 速度粗筛")
    print("=" * 60)

    path = os.path.join(RESULT_DIR, "cleaned_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"请先运行 data_clean.py：{path}")

    df = pd.read_csv(path, dtype={"imsi": str, "time": str,
                                  "lac_id": str, "cell_id": str})
    df = df.sort_values(["imsi", "time"]).reset_index(drop=True)
    print(f"[读取] {len(df)} 行, {df['imsi'].nunique()} 用户")

    # 步骤1 低频点过滤
    df["laci"] = df["lac_id"] + "-" + df["cell_id"]
    df["laci_count"] = df.groupby(["imsi", "laci"])["laci"].transform("count")
    df["is_low_freq"] = df["laci_count"] < LOW_FREQ_THRESHOLD
    print(f"[步骤1] 低频点：{df['is_low_freq'].sum()} "
          f"({df['is_low_freq'].mean()*100:.1f}%)")

    # 步骤2 连续段识别
    df["user_change"] = df["imsi"] != df["imsi"].shift(1)
    df["prev_low_freq"] = df["is_low_freq"].shift(1).fillna(True)
    df["new_segment"] = df["user_change"] | (
        (~df["is_low_freq"]) & df["prev_low_freq"])
    df["segment_id"] = df.groupby("imsi")["new_segment"].cumsum()
    df.loc[df["is_low_freq"], "segment_id"] = -1

    seg_count = df[df["segment_id"] != -1].groupby(
        ["imsi", "segment_id"]).ngroups
    print(f"[步骤2] 连续段：{seg_count} 段")

    # 步骤3 时间差
    df["time_dt"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S",
                                   errors="coerce")
    df["time_diff"] = df["time_dt"].diff().dt.total_seconds()
    print(f"[步骤3] 时间差中位数：{df['time_diff'].median():.0f} 秒")

    # 步骤4 距离差（shift + Haversine）
    df["prev_lon"] = df["lon"].shift(1)
    df["prev_lat"] = df["lat"].shift(1)
    df["dist_diff"] = haversine(df["prev_lon"], df["prev_lat"],
                                 df["lon"], df["lat"], unit="km")
    print(f"[步骤4] 距离差中位数：{df['dist_diff'].median():.4f} km")

    # 步骤5 边界处理
    df.loc[df["user_change"], ["time_diff", "dist_diff"]] = np.nan
    df["segment_change"] = (df["segment_id"] != df["segment_id"].shift(1)) | \
                            (df["segment_id"] == -1)
    df.loc[df["segment_change"], ["time_diff", "dist_diff"]] = np.nan

    # 步骤6 速度计算
    df["speed"] = df["dist_diff"] * 3600 / df["time_diff"]
    print(f"[步骤6] 速度中位数：{df['speed'].median():.2f} km/h")

    # 步骤7 边界情况处理
    zero_time = df["time_diff"] == 0
    abnormal = df["speed"] > ABNORMAL_SPEED_THRESHOLD
    print(f"[步骤7] 时间差为0：{zero_time.sum()}，"
          f"速度>{ABNORMAL_SPEED_THRESHOLD}：{abnormal.sum()}")
    df.loc[zero_time, "speed"] = np.nan
    df.loc[abnormal, "speed"] = np.nan

    # 步骤8 出行点判定
    df["is_travel_stage1"] = df["speed"] > TRAVEL_SPEED_THRESHOLD
    n_travel = df["is_travel_stage1"].sum()
    print(f"[步骤8] 阶段1出行点：{n_travel} ({n_travel/len(df)*100:.1f}%)")

    # 步骤9 段级统计
    valid = df[df["segment_id"] != -1]
    seg_stats = valid.groupby(["imsi", "segment_id"]).agg(
        points=("lon", "count"),
        max_speed=("speed", "max"),
        mean_speed=("speed", "mean"),
        start_time=("time", "min"),
        end_time=("time", "max"),
    ).reset_index()
    seg_stats["is_travel_segment"] = seg_stats["max_speed"] > TRAVEL_SPEED_THRESHOLD
    print(f"[步骤9] 段总数 {len(seg_stats)}，"
          f"出行段 {seg_stats['is_travel_segment'].sum()}")

    # 输出
    travel_out = df[df["is_travel_stage1"]][
        ["imsi", "time", "lac_id", "cell_id", "lon", "lat", "speed"]]
    travel_out.to_csv(os.path.join(RESULT_DIR, "stage1_travel_points.csv"),
                      index=False, encoding="utf-8")
    seg_stats.to_csv(os.path.join(RESULT_DIR, "stage1_segments.csv"),
                     index=False, encoding="utf-8")
    df.to_csv(os.path.join(RESULT_DIR, "stage1_full.csv"),
              index=False, encoding="utf-8")
    print(f"[输出] stage1_travel_points.csv / stage1_segments.csv / stage1_full.csv")
    return df


if __name__ == "__main__":
    stage1()