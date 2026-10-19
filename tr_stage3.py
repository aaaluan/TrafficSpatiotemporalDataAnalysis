import os
import pandas as pd
import numpy as np
from config import RESULT_DIR, DIST_THRESHOLD, TIME_THRESHOLD
from utils import haversine


def stage3():
    print("=" * 60)
    print("阶段3：规则精判与标签输出")
    print("=" * 60)

    # 读取阶段2结果
    cluster_path = os.path.join(RESULT_DIR, "stage2_cluster.csv")
    if not os.path.exists(cluster_path):
        print("错误：请先运行 tr_stage2.py")
        return None
    df = pd.read_csv(cluster_path, dtype={"imsi": str, "time": str,
                                          "lac_id": str, "cell_id": str})
    print(f"数据: {len(df)} 行, {df['imsi'].nunique()} 用户")

    # 初始化标签为出行(1)
    df["label"] = 1
    df["dist_to_center"] = np.nan
    df["time_span_hours"] = np.nan

    # 候选驻留点（非阶段1+阶段2出行点，且有簇标签）
    candidates = df[(~df["is_travel_stage12"]) & (df["cluster_id"] >= 0)]
    print(f"候选驻留点: {len(candidates)}")

    # 按 imsi-簇ID 分组
    groups = candidates.groupby(["imsi", "cluster_id"])
    print(f"组数: {len(groups)}")

    stay_count = 0
    travel_count = 0
    processed_groups = 0

    for (imsi, cid), group in groups:
        # 按时间排序（保留原始 df 索引，用于写回）
        group = group.sort_values("time")
        orig_index = group.index
        # 簇中心
        center_lon = group["lon"].mean()
        center_lat = group["lat"].mean()
        # 距簇中心距离（Haversine，米）
        dists = haversine(group["lon"], group["lat"],
                          center_lon, center_lat, unit="m").to_numpy()
        # 与下一个点的时间跨度（小时）
        time_dt = pd.to_datetime(group["time"], format="%Y%m%d%H%M%S", errors="coerce")
        next_time = time_dt.shift(-1)
        time_spans = ((next_time - time_dt).dt.total_seconds() / 3600).to_numpy()
        # 判定：距离<阈值 且 时间跨度<阈值 → 驻留(0)，否则出行(1)
        is_stay = (dists < DIST_THRESHOLD) & (time_spans < TIME_THRESHOLD)
        # 写回主 DataFrame
        df.loc[orig_index, "dist_to_center"] = dists
        df.loc[orig_index, "time_span_hours"] = time_spans
        df.loc[orig_index[is_stay], "label"] = 0
        stay_count += int(is_stay.sum())
        travel_count += int((~is_stay).sum())
        processed_groups += 1
        if processed_groups % 500 == 0:
            print(f"  已处理 {processed_groups}/{len(groups)} 组...")

    # 统计
    total = stay_count + travel_count
    print(f"\n阶段3精判结果:")
    print(f"  处理组数: {processed_groups}")
    print(f"  驻留(0): {stay_count} ({stay_count/total*100:.1f}%)")
    print(f"  出行(1): {travel_count} ({travel_count/total*100:.1f}%)")
    print(f"\n最终标签分布:")
    print(f"  驻留(0): {(df['label'] == 0).sum()} "
          f"({(df['label'] == 0).mean()*100:.1f}%)")
    print(f"  出行(1): {(df['label'] == 1).sum()} "
          f"({(df['label'] == 1).mean()*100:.1f}%)")

    # 输出
    output_path = os.path.join(RESULT_DIR, "stage3_label.csv")
    output_df = df[["imsi", "time", "lac_id", "cell_id", "lon", "lat", "label"]].copy()
    output_df = output_df.sort_values(["imsi", "time"]).reset_index(drop=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n输出: {output_path}")
    print(f"  行数: {len(output_df)}")

    # 同时保存完整结果（含中间字段，供分析使用）
    full_path = os.path.join(RESULT_DIR, "stage3_full.csv")
    df.to_csv(full_path, index=False, encoding="utf-8")
    print(f"完整结果: {full_path}")

    print("\n" + "=" * 60)
    print("阶段3完成！")
    print("=" * 60)
    return df


if __name__ == "__main__":
    stage3()
