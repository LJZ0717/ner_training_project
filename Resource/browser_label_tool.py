import streamlit as st
import pandas as pd
import os

# 📂 CSV 路径
CSV_FILE = "section_labels.csv"


# 📦 加载数据
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    st.error(f"⚠️ 找不到文件：{CSV_FILE}")
    st.stop()

# 找到未标注的行
unlabeled_df = df[df['label'].isnull() | (df['label'] == '')]

if unlabeled_df.empty:
    st.success("🎉 所有段落已标注完成！")
else:
    st.title("📑 Section Labeling Tool")
    st.write("请为以下段落选择标签：")

    index = st.session_state.get('index', 0)
    text = unlabeled_df.iloc[index]['text']
    st.text_area("段落内容", text, height=200)

    col1, col2, col3 = st.columns(3)
    if col1.button("✅ 感兴趣 (1)"):
        df.loc[unlabeled_df.index[index], 'label'] = 1
        index += 1
    if col2.button("❌ 不感兴趣 (0)"):
        df.loc[unlabeled_df.index[index], 'label'] = 0
        index += 1
    if col3.button("⏭️ 跳过"):
        index += 1

    # 保存进度
    try:
        df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    except PermissionError:
        st.warning("⚠️ 文件被占用，已保存为 section_labels_updated.csv")
        df.to_csv("section_labels_updated.csv", index=False, encoding="utf-8-sig")

    # 更新索引
    st.session_state['index'] = index

    st.info(f"🔖 已标注 {len(df) - len(unlabeled_df) + index} / {len(df)} 条")

    if index >= len(unlabeled_df):
        st.success("🎉 所有段落已标注完成！")
