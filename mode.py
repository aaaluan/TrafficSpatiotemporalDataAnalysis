"""
出行方式识别
1.识别出行段
2.判定速度规则
3.读取静态出行方式数据
4.空间匹配
"""
import pandas as pd
import numpy as np
import os
from config import RESULT_DIR,DATA_DIR,MATCH_THRESHOLD,SUBWAY_MIN_POINTS,BUS_MIN_POINTS,SPEED_OTHER ,SPEED_BUS
from utils import haversine,Timer

def mode_analysis():
    df = pd.read_csv(os.path.join(RESULT_DIR,"stage3_label.csv"),
                     dtype = {"imsi":str,"time":str,"lac_id":str,"cell_id":str}
                     )
    print(f"[mode]阶段三结果：{len(df)}行，{df['imsi'].nunique()}用户")
    print(f"[mode]驻留点：{(df['label'] == 0).sum()}")
    print(f"[mode]出行点：{(df['label'] == 1).sum()}")

    df["is_travel"] = df["label"] == 1
    print(f"[mode]出行点：{df['is_travel'].sum()} ({df['is_travel'].mean()*100:.1f}%)")

    df["user_change"] = df["imsi"] != df["imsi"].shift(1)
    df["travel_change"] = df["is_travel"] != df["is_travel"].shift(1)
    df["seg_change"] = df["user_change"] | df["travel_change"]
    df["seg_id"] = df.groupby("imsi")["seg_change"].cumsum()

    travel_df = df[df["is_travel"]].copy()
    print(f"[mode]出行段数（含单点段）：{travel_df.groupby(['imsi','seg_id']).ngroups}")

    # =========关键优化：设置多重索引，避免循环内布尔大表查询 =========
    travel_df = travel_df.set_index(["imsi","seg_id"])

    segments = travel_df.groupby(["imsi","seg_id"]).agg(
        start_time = ("time","min"),
        end_time = ("time",max),
        points = ("time","count"),
        start_lon = ("lon","first"),
        start_lat = ("lat","first"),
        end_lon = ("lon","last"),
        end_lat = ("lat","last")
    ).reset_index()

    segments["start_dt"] = pd.to_datetime(segments["start_time"],format = "%Y%m%d%H%M%S",errors = "coerce")
    segments["end_dt"] = pd.to_datetime(segments["end_time"],format = "%Y%m%d%H%M%S",errors = "coerce")
    segments["duration_hours"] = (segments["end_dt"] - segments["start_dt"]).dt.total_seconds() / 3600.0

    avg_speeds = []
    seg_count_total = len(segments)
    with Timer("计算出行段的平均速度耗时："):
        for loop_idx, (idx,seg) in enumerate(segments.iterrows()):
            # 【优化】索引直接切片，代替 travel_df[...]布尔过滤
            try:
                seg_points = travel_df.loc[(seg["imsi"], seg["seg_id"])].copy()
            except KeyError:
                avg_speeds.append(np.nan)
                continue
            seg_points = seg_points.sort_values("time").reset_index()

            if len(seg_points) < 2:
                avg_speeds.append(np.nan)
                continue

            seg_points["time_dt"] = pd.to_datetime(seg_points["time"],format = "%Y%m%d%H%M%S",errors = "coerce")
            seg_points["time_diff"] = seg_points["time_dt"].diff().dt.total_seconds()
            seg_points["prev_lon"] = seg_points["lon"].shift(1)
            seg_points["prev_lat"] = seg_points["lat"].shift(1)
            seg_points["dist_diff"] = haversine(seg_points["lon"],seg_points["lat"],seg_points["prev_lon"],seg_points["prev_lat"])
            seg_points["speed"] = seg_points["dist_diff"]* 3600 / seg_points["time_diff"]

            valid_speeds = seg_points["speed"][(seg_points["speed"] > 0) & (seg_points["speed"] < 200)]
            if len(valid_speeds) > 0:
                avg_speeds.append(valid_speeds.mean())
            else:
                avg_speeds.append(np.nan)

            # 打印进度，观察程序是否在跑
            if (loop_idx+1) % 50 == 0:
                print(f"[mode]速度计算进度 {loop_idx+1}/{seg_count_total}")

        segments["avg_speeds"] = avg_speeds
        segments = segments[segments["avg_speeds"].notna()].copy()
        print(f"[mode]有效出行段：{len(segments)}")
        print(f"[mode] 平均速度：{segments['avg_speeds'].mean():.2f}km/h")

        before = len(segments)
        segments = segments[segments["points"] >= 2].copy()
        after = len(segments)
        print(f"[mode]过滤点<2的出行段：{before} -> {after} (删除{before - after})")

    segments["mode_by_speed"] = segments["avg_speeds"].apply(judge_mode_by_speed)
    print(f"[mode]速度规则判定：")
    print(segments["mode_by_speed"].value_counts().to_string())

    # 读取出行方式静态数据
    mode_path = os.path.join(DATA_DIR,"出行方式静态数据.csv")
    mode_df = pd.read_csv(mode_path,encoding = "gbk",header = None)
    print(f"\n[mode]出行方式静态数据：{len(mode_df)}行")
    mode_df.columns = ["lon","lat","mode_type","station_name","line_name"]
    mode_df["lon"] = pd.to_numeric(mode_df["lon"],errors = "coerce")
    mode_df["lat"] = pd.to_numeric(mode_df["lat"],errors = "coerce")
    mode_df = mode_df.dropna(subset= ["lon","lat"]).drop_duplicates()

    subway_df = mode_df[mode_df["mode_type"].str.contains("地铁", na = False)].copy()
    bus_df = mode_df[mode_df["mode_type"].str.contains("公交",na = False)].copy()
    print(f"[mode]地铁站点{len(subway_df)},公交站点{len(bus_df)}")

    from scipy.spatial import KDTree
    subway_tree = KDTree(np.column_stack([subway_df["lon"], subway_df["lat"]]))
    bus_tree = KDTree(np.column_stack([bus_df["lon"], bus_df["lat"]]))
    subway_match_list = []
    bus_match_list = []
    seg_total = len(segments)

    for loop_idx, (idx,seg) in enumerate(segments.iterrows()):
        try:
            seg_points = travel_df.loc[(seg["imsi"], seg["seg_id"])].copy()
        except KeyError:
            subway_match_list.append(0)
            bus_match_list.append(0)
            continue
        seg_points = seg_points.sort_values("time").reset_index()
        pts_xy = seg_points[["lon","lat"]].to_numpy()

        d_sub,_ = subway_tree.query(pts_xy, k=1)
        cnt_sub = np.sum(d_sub < MATCH_THRESHOLD/111000)

        d_bus,_ = bus_tree.query(pts_xy, k=1)
        cnt_bus = np.sum(d_bus < MATCH_THRESHOLD/111000)

        subway_match_list.append(int(cnt_sub))
        bus_match_list.append(int(cnt_bus))

        if (loop_idx+1) % 50 == 0:
            print(f"[mode]站点匹配进度 {loop_idx+1}/{seg_total}")

    segments["subway_match_points"] = subway_match_list
    segments["bus_match_points"] = bus_match_list

    segments["final_mode"] = segments.apply(final_judge,axis = 1)
    print("[mode]最终出行方式：")
    print(segments["final_mode"].value_counts().to_string())

    output_path = os.path.join(RESULT_DIR,"mode_result.csv")
    output_cols = ['imsi','seg_id','start_time','end_time','duration_hours','points',
                   'start_lon','start_lat','end_lon','end_lat',
                   'avg_speeds','mode_by_speed',
                   'subway_match_points','bus_match_points','final_mode'
                   ]
    segments[output_cols].to_csv(output_path,index = False,encoding = "utf-8-sig")
    print(f"\n输出：{output_path}")
    print(f" 行数{len(segments)}")
    return segments

def judge_mode_by_speed(avg_speed):
    if avg_speed < 12:
        return "其他"
    if avg_speed <= 20:
        return "公交"
    else:
        return "地铁或小汽车"

def final_judge(row):
    if row["subway_match_points"] >= SUBWAY_MIN_POINTS:
        return "地铁"
    if row["bus_match_points"] >= BUS_MIN_POINTS:
        return "公交"
    if row['avg_speeds'] > SPEED_BUS:
        return "小汽车"
    else:
        return "其他"

if __name__ == "__main__":
    mode_analysis()
