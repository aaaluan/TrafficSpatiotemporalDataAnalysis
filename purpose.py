import pandas as pd
import numpy as np
import os

from config import LIVING_DURATION_THRESHOLD,PURPOSE_MATCH_THRESHOLD
from utils import parse_time_to_dt,haversine,Timer
from config import DATA_DIR, RESULT_DIR

#  模块三 识别出行目的

# 切驻留段
def purpose_analysis():
    label_path = os.path.join(RESULT_DIR,"stage3_label.csv")
    if not os.path.exists(label_path):
        print("错误：请先运行tr_stage3.py")
        return None
    df = pd.read_csv(label_path,dtype = {"imsi":str,
                                         "lac_id":str,
                                         "cell":str,
                                         "time":str
                                         })
    print(f" 数据：{len(df)}行，{df["imsi"].nunique()}用户")
# 驻留段识别
    df["time_dt"] = parse_time_to_dt(df["time"])
    df = df.sort_values(["imsi", "time"]).reset_index(drop=True)

    user_change = (df["imsi"] != df["imsi"].shift(1)).astype(int)  # 用户切换
    break_by_travel = (df["label"] == 1).astype(int)  # 驻留切换
    df["stay_seg"] = (user_change + break_by_travel).groupby(df["imsi"]).cumsum()  # 生成驻留段id

    stay = df[df["label"] == 0]  #  只保留驻留点

    #驻留段特征聚合
    agg = stay.groupby(["imsi", "stay_seg"]).agg(
        start_time=('time_dt', 'min'),
        end_time=('time_dt', 'max'),
        points=('time', 'count'),
        center_lon=('lon', 'mean'),  # center_lon\center_lat 段内点坐标均值
        center_lat=('lat', 'mean'),
    ).reset_index()
    agg["duration_hours"] = (agg["end_time"] - agg["start_time"]).dt.total_seconds() / 3600.0  # 计算停留时间
    agg["start_hour"] = agg["start_time"].dt.hour  # 提取开始时的小时
    agg["end_hour"] = agg["end_time"].dt.hour  # 提取结束时的小时
    print(f"  驻留段数: {len(agg)}")

    # 时间规则判定结果
    agg["purpose_by_time"] = agg.apply(lambda row:decide_purpose(row),axis = 1)
    print(f"时间规则判定结果：")
    print(agg["purpose_by_time"].value_counts().to_string())

    # 读取静态POI数据并清洗
    poi = pd.read_csv(os.path.join(DATA_DIR, "出行目的静态数据.csv"), encoding="gbk",
                      names=["lon", "lat", "poi_type", "poi_name"]
                          )
    poi["lon"] = pd.to_numeric(poi["lon"], errors="coerce")
    poi["lat"] = pd.to_numeric(poi["lat"], errors="coerce")
    poi = poi.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    # 广播算”段中心 x 全部POI“距离矩阵
    poi_lon,poi_lat = poi["lon"].to_numpy()[None,:],poi["lat"].to_numpy()[None,:]
    c_lon = agg["center_lon"].to_numpy()[:,None]  # [:,None] ：将一维数组变成两维，变成单列的二维数组，方便后续计算距离
    c_lat = agg["center_lat"].to_numpy()[:,None]

    dist_m = haversine(c_lon,c_lat,poi_lon,poi_lat,unit = 'm')  # 停留中心行与POI列构成的矩阵
    nearest_idx = dist_m.argmin(axis = 1)  #  选出每个停留中心到POI最近的距离的列索引
    nearest_dist = dist_m[np.arange(len(agg)),nearest_idx]
    agg["nearest_poi_name"] = poi.loc[nearest_idx,"poi_name"].to_numpy()
    agg["nearest_poi_type"] = poi.loc[nearest_idx,"poi_type"].to_numpy()
    agg["nearest_dist_m"] = np.round(nearest_dist,1)
    agg["match_success"] = nearest_dist < PURPOSE_MATCH_THRESHOLD
    print(f"POI空间匹配成功: {agg['match_success'].astype(int).sum() / len(agg['match_success'])*100:.1f}%(阈值1000m)")


    with Timer("出行目的识别 "):
      agg["final_purpose"] = agg.apply(decide_purpose,axis =1)
      print(f"出行目的的分布：{agg['final_purpose'].value_counts()}")

    # 输出
    out_path = os.path.join(RESULT_DIR,"purpose_result.csv")
    output_cols = ['imsi', 'start_time', 'end_time','duration_hours' ,'points', 'center_lon','center_lat',
                   'purpose_by_time', 'nearest_poi_name', 'nearest_poi_type', 'nearest_dist_m',
                   'match_success', 'final_purpose']
    agg[output_cols].to_csv(out_path,index = False,encoding = 'utf-8')
    print(f"\n输出：{out_path}")
    print("出行目的识别完成！")
    return agg
#  夜间判定
def is_night(hour):
  return (0 <= hour <= 6) or (20 <= hour <= 23)



def decide_purpose(row):
    s, e, dur = row["start_hour"], row["end_hour"], row["duration_hours"]
    # 1. 居住：长时间且端点在夜间
    if dur > LIVING_DURATION_THRESHOLD and (is_night(s) or is_night(e)):
        return "居住"

    #  2. 空间证据：1000m内最近POI是购物/旅游，直接采用
    poi_type = row.get("nearest_poi_type")
    if bool(row.get("match_success", False)) and poi_type in ("购物", "旅游"):
        return poi_type

    #  3. 时间兜底：先窄后宽
    if s >= 9 and e <= 17:
        return "旅游"
    if s >= 9 and e <= 21:
        return "购物"

    return "其他"



if __name__ == "__main__":
    purpose_analysis()

