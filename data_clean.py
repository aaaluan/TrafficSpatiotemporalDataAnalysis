import os
import pandas as pd
import numpy as np
from config import (DATA_DIR,RESULT_DIR,RAW_FILE, BS_FILE,
                    RAW_COLS,TARGET_DATE,MIN_RECORDS_PER_USER,
                    SAMPLE_ROWS,LON_MAX,LON_MIN,LAT_MAX,LAT_MIN)

def _ms_to_yyyymmddhhmmss(ms_series):
    ms = pd.to_numeric(ms_series, errors="coerce")
    dt_utc = pd.to_datetime(ms, unit="ms", utc=True, errors="coerce")
    dt_bj = dt_utc.dt.tz_convert("Asia/Shanghai")
    return dt_bj.dt.strftime("%Y%m%d%H%M%S")

def clean_data():
    print("="*60)
    print("模块一 数据清洗")
    print("="*60)

    # 读取原始数据
    raw_path = os.path.join(DATA_DIR,RAW_FILE)
    bs_path = os.path.join(DATA_DIR,BS_FILE)

    dtype_map = {"imsi":str,"lac_id":str,"cell_id":str,
                 "phone":str,"time":str}
    if SAMPLE_ROWS:
        df = pd.read_csv(raw_path,header=None,names=RAW_COLS,dtype=dtype_map,
                         nrows=SAMPLE_ROWS,encoding="utf-8")
        print(f"[读取]抽样{SAMPLE_ROWS}行")
    else:
        df = pd.read_csv(raw_path, header=None, names=RAW_COLS,
                         dtype=dtype_map, encoding="utf-8")
        print(f"[读取] 原始数据 {len(df)} 行")
    df_bs = pd.read_csv(bs_path, header=None,
                        names=["longitude", "latitude", "laci"],
                        dtype={"laci": str}, encoding="utf-8")
    print(f"[读取] 基站数据 {len(df_bs)} 行")

    # 规则1 字段抽取
    df["time"] = _ms_to_yyyymmddhhmmss(df["timestamp"])
    df = df[["imsi","time","lac_id","cell_id"]].copy()
    print(f"[规则1]字段抽取: {len(df)}行")

    # 规则2 去脏imsi
    before = len(df)
    dirty = df["imsi"].str.contains(r"[#* ^\s]",na=False)
    df = df[~dirty].copy()
    print(f"[规则2] 去脏imsi：删 {before - len(df)} 行 → {len(df)} 行")

    # 规则3 去空lac_id/cell_id
    before = len(df)
    df["lac_id"] = df["lac_id"].fillna("").str.strip()
    df["cell_id"] = df["cell_id"].fillna("").str.strip()
    df = df[(df["lac_id"]!="")&(df["cell_id"]!="")].copy()
    print(f"[规则3] 去空lac/cell：删 {before - len(df)} 行 → {len(df)} 行")

    # 规则4 时间格式转换
    print(f"[规则4] 时间格式,样例 time={df['time'].iloc[0]}")

    # 规则5 过滤目标日期
    before = len(df)
    df["date"] = df["time"].str[:8]
    df = df[df["date"]==TARGET_DATE].copy()
    print(f"[规则5] 过滤日期{TARGET_DATE}：删 {before - len(df)} 行 → {len(df)} 行")

    # 规则6 基站关联补经纬度
    df["laci"] = df["lac_id"]+"-"+df["cell_id"]
    before = len(df)
    df = pd.merge(df,df_bs,how="left")
    print(f"[规则6] 基站关联：{len(df)} 行（关联前 {before}）")
    no_geo = df["longitude"].isna()|df["latitude"].isna()
    print(f"未匹配基站:{no_geo.sum()}行")
    df = df.dropna(subset=["longitude","latitude"].copy())
    df = df.rename(columns={"longitude": "lon", "latitude": "lat"})

    df = df[(df["lon"].between(LON_MIN, LON_MAX)) &
            (df["lat"].between(LAT_MIN, LAT_MAX))].copy()
    print(f"[规则6] 沈阳范围过滤 → {len(df)} 行")

    # 规则7 按imsi + time 排序
    df = df.sort_values(["imsi","time"]).reset_index(drop=True)
    print(f"[规则7] 排序完成")

    # 规则8 分钟去重
    before = len(df)
    df["minute"] = df.time.str[:12]
    df = df.drop_duplicates(subset=["imsi","minute"],keep="first").copy()
    print(f"[规则8] 分钟去重：删 {before - len(df)} 行 → {len(df)} 行")

    # 规则9 过滤低频用户
    before_rows, before_users = len(df), df["imsi"].nunique()
    df = df.groupby("imsi").filter(lambda x: len(x) >= MIN_RECORDS_PER_USER).copy()
    print(f"[规则9] 过滤低频用户(<{MIN_RECORDS_PER_USER})：")
    print(f"{before_users}用户/{before_rows}行 → "
          f"{df['imsi'].nunique()}用户/{len(df)}行")

    # 输出
    out_cols = ["imsi", "time", "lac_id", "cell_id", "lon", "lat"]
    out = df[out_cols].reset_index(drop=True)
    out_path = os.path.join(RESULT_DIR, "cleaned_data.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[输出] {out_path}  {len(out)} 行")
    print(f"[验证] 日期={out['time'].str[:8].unique().tolist()}")
    print(f"[验证] 经度范围=[{out['lon'].min():.4f}, {out['lon'].max():.4f}]")
    print(f"[验证] 纬度范围=[{out['lat'].min():.4f}, {out['lat'].max():.4f}]")
    return out

if __name__ == "__main__":
    clean_data()