import os
import pandas as pd
import numpy as np
from config import (RESULT_DIR, K_MIN, K_MAX, RANDOM_STATE,
                    KMEANS_MAX_ITER, KMEANS_TOL)


def kmeans(X, k, max_iter=KMEANS_MAX_ITER, tol=KMEANS_TOL, random_state=RANDOM_STATE):
    """手写 k-means 聚类"""
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    # 初始化：随机选择 k 个不重复的点作为初始中心
    indices = rng.choice(n, k, replace=False)
    centers = X[indices].copy()
    # 迭代：分配 -> 更新 -> 检查收敛
    for iteration in range(max_iter):
        diff = X[:, None, :] - centers[None, :, :]
        dist_sq = (diff ** 2).sum(axis=2)
        labels = dist_sq.argmin(axis=1)
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                new_centers[i] = X[mask].mean(axis=0)
            else:
                # 空簇：随机选择一个点作为新中心
                new_centers[i] = X[rng.integers(n)]
        center_shift = np.linalg.norm(new_centers - centers, axis=1).max()
        centers = new_centers
        if center_shift < tol:
            break
    # 计算最终 SSE
    diff = X[:, None, :] - centers[None, :, :]
    dist_sq = (diff ** 2).sum(axis=2)
    labels = dist_sq.argmin(axis=1)
    sse = dist_sq.min(axis=1).sum()
    return labels, centers, sse


def calc_dunn_index(X, labels):
    """计算邓恩指数：簇间最小距离 / 簇内最大距离（越大越好）"""
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    diff = X[:, None, :] - X[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    # 簇内最大距离（簇直径）
    max_intra = 0.0
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 1:
            max_intra = max(max_intra, dist_matrix[np.ix_(mask, mask)].max())
    # 簇间最小距离
    min_inter = np.inf
    for i in range(k):
        for j in range(i + 1, k):
            mask_i = labels == unique_labels[i]
            mask_j = labels == unique_labels[j]
            if mask_i.sum() > 0 and mask_j.sum() > 0:
                min_inter = min(min_inter, dist_matrix[np.ix_(mask_i, mask_j)].min())
    if max_intra == 0 or min_inter == np.inf:
        return 0.0
    return min_inter / max_intra


def select_best_k(X, k_min=K_MIN, k_max=K_MAX, random_state=RANDOM_STATE):
    """遍历 K=k_min~k_max，用邓恩指数选最优 K"""
    n = X.shape[0]
    # 关键：K 上限不能超过点数，否则 rng.choice 报错
    actual_k_max = min(k_max, n)
    if actual_k_max < k_min:
        labels = np.zeros(n, dtype=int)
        centers = X.mean(axis=0, keepdims=True)
        return 1, labels, centers, 0.0
    best_k = k_min
    best_dunn = -1.0
    best_labels = None
    best_centers = None
    for k in range(k_min, actual_k_max + 1):
        labels, centers, sse = kmeans(X, k, random_state=random_state)
        dunn = calc_dunn_index(X, labels)
        if dunn > best_dunn:
            best_dunn = dunn
            best_k = k
            best_labels = labels
            best_centers = centers
    return best_k, best_labels, best_centers, best_dunn


def stage2():
    """执行阶段2完整流程"""
    print("=" * 60)
    print("阶段2：k-means聚类与最优K选择")
    print("=" * 60)

    # 读取阶段1数据
    full_path = os.path.join(RESULT_DIR, "stage1_full.csv")
    if not os.path.exists(full_path):
        print("错误：请先运行 tr_stage1.py")
        return None
    df = pd.read_csv(full_path, dtype={"imsi": str, "time": str,
                                       "lac_id": str, "cell_id": str, "laci": str})
    print(f"数据: {len(df)} 行, {df['imsi'].nunique()} 用户")

    # 初始化阶段2标签
    df["cluster_id"] = -1
    df["best_k"] = 0
    df["is_singleton"] = False
    df["is_travel_stage2"] = False

    total_singleton = 0
    total_clustered = 0
    user_best_k = []
    processed_users = 0

    # 对每个用户聚类
    for imsi, user_df in df.groupby("imsi"):
        # 排除阶段1出行点（阶段2只对非出行点聚类）
        non_travel = user_df[~user_df["is_travel_stage1"]]
        if len(non_travel) < 2:
            continue
        X = non_travel[["lon", "lat"]].values
        best_k, labels, centers, best_dunn = select_best_k(X)
        # 记录簇标签
        df.loc[non_travel.index, "cluster_id"] = labels
        df.loc[non_travel.index, "best_k"] = best_k
        total_clustered += len(non_travel)
        user_best_k.append(best_k)
        processed_users += 1
        # 单点成簇判定
        cluster_sizes = pd.Series(labels).value_counts()
        singleton_clusters = cluster_sizes[cluster_sizes == 1].index
        if len(singleton_clusters) > 0:
            is_singleton = np.isin(labels, singleton_clusters)
            df.loc[non_travel.index[is_singleton], "is_singleton"] = True
            df.loc[non_travel.index[is_singleton], "is_travel_stage2"] = True
            total_singleton += is_singleton.sum()

    # 合并阶段1和阶段2出行点
    df["is_travel_stage12"] = df["is_travel_stage1"] | df["is_travel_stage2"]

    # 统计
    print(f"\n聚类统计:")
    print(f"  处理用户数: {processed_users}")
    print(f"  聚类点数: {total_clustered}")
    print(f"  最优K分布:")
    for k in range(1, K_MAX + 1):
        count = user_best_k.count(k)
        if count > 0:
            print(f"    K={k}: {count} 用户 ({count/processed_users*100:.1f}%)")
    print(f"  单点成簇点数: {total_singleton} "
          f"({total_singleton/total_clustered*100:.1f}%)")
    print(f"\n出行点统计:")
    print(f"  阶段1出行点: {df['is_travel_stage1'].sum()}")
    print(f"  阶段2出行点(单点成簇): {df['is_travel_stage2'].sum()}")
    print(f"  合并出行点: {df['is_travel_stage12'].sum()}")
    print(f"  驻留点: {(~df['is_travel_stage12']).sum()}")

    # 输出
    cluster_path = os.path.join(RESULT_DIR, "stage2_cluster.csv")
    df[["imsi", "time", "lac_id", "cell_id", "lon", "lat",
        "is_travel_stage1", "cluster_id", "best_k",
        "is_singleton", "is_travel_stage2", "is_travel_stage12"]
    ].to_csv(cluster_path, index=False, encoding="utf-8")
    print(f"\n聚类结果: {cluster_path}")

    travel_path = os.path.join(RESULT_DIR, "stage2_travel_points.csv")
    df[df["is_travel_stage12"]][
        ["imsi", "time", "lac_id", "cell_id", "lon", "lat"]
    ].to_csv(travel_path, index=False, encoding="utf-8")
    print(f"合并出行点: {travel_path}")

    print("\n" + "=" * 60)
    print("阶段2完成！")
    print("=" * 60)
    return df


if __name__ == "__main__":
    stage2()
