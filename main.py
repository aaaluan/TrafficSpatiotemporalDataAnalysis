# main.py
"""
离线全流程入口，完整流水线顺序：
data_clean → tr_stage1 → tr_stage2 → tr_stage3 → purpose → mode → heatmap实时商圈热力 → visualize全部6张图
输出：所有csv结果 + figures下6张png图片
"""
import os
from config import RESULT_DIR, FIG_DIR
from data_clean import clean_data
from tr_stage1 import stage1
from tr_stage2 import stage2
from tr_stage3 import stage3
from purpose import purpose_analysis
from mode import mode_analysis
from heatmap import heatmap_analysis
from visualize import visualize_all_save


def main():
    print("=" * 75)
    print("交通时空大数据分析系统 离线完整流水线启动")
    print("=" * 75)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 用户轨迹识别流水线
    clean_data()
    stage1()
    stage2()
    stage3()
    purpose_analysis()
    mode_analysis()

    # Day09 实时商圈热力分析（独立模块，读取实时流数据）
    heatmap_analysis()

    # Day10 综合可视化生成全部6张图
    visualize_all_save()

    print("\n" + "=" * 75)
    print("全部模块执行完成！结果目录：", RESULT_DIR)
    print("=" * 75)


if __name__ == "__main__":
    main()
