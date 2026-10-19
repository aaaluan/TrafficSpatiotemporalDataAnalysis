# streamlit_app.py
"""
Web Dashboard，两行三列卡片布局，对应Day10的6张标准图
启动命令：streamlit run streamlit_app.py
前置条件：必须先运行 python main.py 生成所有csv结果文件

"""
import os
import streamlit as st
import pandas as pd
from config import RESULT_DIR, FIG_DIR
from visualize import (
    plot_stage3_pie,
    plot_purpose_bar,
    plot_mode_bar,
    plot_24h_travel_line,
    plot_biz_hour_heatmap,
    plot_top5_3d_traj
)

st.set_page_config(page_title="交通时空大数据分析与可视化系统", layout="wide")
st.title("交通时空大数据分析与可视化系统")
st.markdown("数据源：沈阳2018‑10‑03信令与实时流数据 | 技术栈：Python Pandas Matplotlib | 六大分析模块一键调用，结果可保存PNG")


@st.cache_data(show_spinner="正在加载分析结果CSV……")
def load_all_result():
    df_label = pd.read_csv(os.path.join(RESULT_DIR, "stage3_label.csv"), dtype={"imsi": str})
    df_purpose = pd.read_csv(os.path.join(RESULT_DIR, "purpose_result.csv"))
    df_mode = pd.read_csv(os.path.join(RESULT_DIR, "mode_result.csv"))
    df_heat = pd.read_csv(os.path.join(RESULT_DIR, "heatmap_result.csv"))
    return df_label, df_purpose, df_mode, df_heat


# 文件缺失捕获
try:
    df_label, df_purpose, df_mode, df_heat = load_all_result()
except FileNotFoundError:
    st.error(" 缺少输出CSV文件！请先在终端执行：`python main.py` 完成离线全流程计算。")
    st.stop()

btn_all = st.button("▶一键生成全部6张图表", type="primary")

# 两行三列布局
row1 = st.columns(3)
row2 = st.columns(3)

# =========卡片1：出行/驻留标签饼图 row1[0] =========
with row1[0]:
    st.subheader("出行/驻留标签分布饼图")
    b1, s1 = st.columns([1, 1])
    run1 = b1.button("调用分析", key="k1")
    save1 = s1.button("保存PNG", key="ks1")
    c1 = st.container(height=360)
    fig1 = None
    with c1:
        if run1 or btn_all:
            fig1 = plot_stage3_pie(df_label)
            st.pyplot(fig1)
        else:
            st.info("点击【调用分析】生成图表")
    if save1 and fig1 is not None:
        fig1.savefig(os.path.join(FIG_DIR, "fig_stage3_pie.png"), dpi=150, bbox_inches="tight")
        st.success("已保存PNG")

# =========卡片2：出行目的分布柱状图 row1[1] =========
with row1[1]:
    st.subheader("出行目的分布柱状图")
    b2, s2 = st.columns([1, 1])
    run2 = b2.button("调用分析", key="k2")
    save2 = s2.button("保存PNG", key="ks2")
    c2 = st.container(height=360)
    fig2 = None
    with c2:
        if run2 or btn_all:
            fig2 = plot_purpose_bar(df_purpose)
            st.pyplot(fig2)
        else:
            st.info("点击【调用分析】生成图表")
    if save2 and fig2 is not None:
        fig2.savefig(os.path.join(FIG_DIR, "fig_purpose_bar.png"), dpi=150, bbox_inches="tight")
        st.success("已保存PNG")

# =========卡片3：24小时出行人次折线图 row1[2] =========
with row1[2]:
    st.subheader("24小时出行人次折线图")
    b3, s3 = st.columns([1, 1])
    run3 = b3.button("调用分析", key="k3")
    save3 = s3.button("保存PNG", key="ks3")
    c3 = st.container(height=360)
    fig3 = None
    with c3:
        if run3 or btn_all:
            fig3 = plot_24h_travel_line(df_label)
            st.pyplot(fig3)
        else:
            st.info("点击【调用分析】生成图表")
    if save3 and fig3 is not None:
        fig3.savefig(os.path.join(FIG_DIR, "fig_hour_line.png"), dpi=150, bbox_inches="tight")
        st.success("已保存PNG")

# =========卡片4：出行方式分布柱状图 row2[0] =========
with row2[0]:
    st.subheader("出行方式分布柱状图")
    b4, s4 = st.columns([1, 1])
    run4 = b4.button("调用分析", key="k4")
    save4 = s4.button("保存PNG", key="ks4")
    c4 = st.container(height=360)
    fig4 = None
    with c4:
        if run4 or btn_all:
            fig4 = plot_mode_bar(df_mode)
            st.pyplot(fig4)
        else:
            st.info("点击【调用分析】生成图表")
    if save4 and fig4 is not None:
        fig4.savefig(os.path.join(FIG_DIR, "fig_mode_bar.png"), dpi=150, bbox_inches="tight")
        st.success("已保存PNG")

# =========卡片5：【课件标准】Top10商圈×24h热力图 row2[1] =========
with row2[1]:
    st.subheader("Top10商圈每小时人次热力图")
    b5, s5 = st.columns([1, 1])
    run5 = b5.button("调用分析", key="k5")
    save5 = s5.button("保存PNG", key="ks5")
    c5 = st.container(height=360)
    fig5 = None
    with c5:
        if run5 or btn_all:
            fig5 = plot_biz_hour_heatmap(df_heat)
            st.pyplot(fig5)
        else:
            st.info("点击【调用分析】生成商圈时间热力图")
    if save5 and fig5 is not None:
        fig5.savefig(os.path.join(FIG_DIR, "fig_heatmap.png"), dpi=150, bbox_inches="tight")
        st.success("已保存PNG")

# =========卡片6：Top5用户3D轨迹图 row2[2] =========
with row2[2]:
    st.subheader("Top5用户3D移动轨迹图")
    b6, s6 = st.columns([1, 1])
    run6 = b6.button("调用分析", key="k6")
    save6 = s6.button("保存PNG", key="ks6")
    c6 = st.container(height=360)
    fig6 = None
    with c6:
        if run6 or btn_all:
            fig6 = plot_top5_3d_traj(df_label)
            st.pyplot(fig6)
        else:
            st.info("点击【调用分析】生成3D轨迹")
    if save6 and fig6 is not None:
        fig6.savefig(os.path.join(FIG_DIR, "fig_3d_traj.png"), dpi=150, bbox_inches="tight")
        st.success("已保存PNG")
