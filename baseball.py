#!pip install streamlit
# !pip install -U beautifulsoup4
import streamlit as st
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# 係数と切片の定義
coef = np.array([26.45187647, 39.93772468, 78.13839736, 61.69789698, 3.72280363,
                 -14.90910587, 23.29650394, -11.9513952, 23.68076343])
intercept = -5.479172400661533

def predict_linear_regression(x):
    return intercept + np.dot(coef, x)

def process_player_stats(player_stats_str):
    player_stats_str = player_stats_str.replace('.', '')
    numbers = [int(num) for num in player_stats_str.split()]
    df = pd.DataFrame([numbers])
    columns = [1, 4, 5, 6, 7, 10, 11, 13, 14, 15, 17, 18, 19, 20]
    df = df.iloc[:, columns]
    df.columns = ["PA", "Hit", "2B", "3B", "HR", "SB", "CS", "SF", "BB", "DB", "DP", "BA", "SLG", "OBP"]
    df["Hit"] = df["Hit"] - df["2B"] - df["3B"] - df["HR"]
    df["Walk"] = df["BB"] - df["DB"]
    df["OPS"] = df["SLG"] + df["OBP"]
    return df

def calculate_event_stats(df):
    event = df[['Hit', '2B', '3B', 'HR', 'SB', 'CS', 'SF', 'DP', 'Walk']].div(df["PA"].iloc[0], axis=0)
    stats = df[["BA", "SLG", "OBP", "OPS"]] / 1000
    return event, stats

def update_random_df(value, df_random):
    i = float(value.item())
    df_random.loc[len(df_random)] = i
    df_random['rank'] = df_random['value'].rank()
    rank = df_random[df_random['value'] == i]['rank'].values[0]
    score = int(rank) / 10
    df_random['rank'] = df_random['value'].rank(ascending=False)
    ranking = df_random[df_random['value'] == i]['rank'].values[0]
    ranking = int(ranking)
    return score, ranking

def process_url_data(url):
    r = requests.get(url)
    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.content, 'html.parser')
    td_tags = soup.find_all('td')
    td_texts = [td.get_text().replace('\u200b', '').replace('\u3000', '') for td in td_tags]
    sublists = [td_texts[k:k+25] for k in range(1, len(td_texts), 25)]

    df = pd.DataFrame(sublists).dropna()
    df = df.iloc[:, [3, 5, 8, 9, 10, 11, 14, 15, 17, 18, 20, 22, 23, 24]].astype(float)
    df.columns = ["BA", "PA", "Hit", "2B", "3B", "HR", "SB", "CS", "SF", "BB", "DB", "DP", "SLG", "OBP"]
    df["OPS"] = df["SLG"] + df["OBP"]
    df["Hit"] = df["Hit"] - df["2B"] - df["3B"] - df["HR"]
    df["Walk"] = df["BB"] - df["DB"]
    df = df.drop(columns=["BB", "DB"])
    df[["Hit", "2B", "3B", "HR", "SB", "CS", "SF", "DP", "Walk"]] = df[["Hit", "2B", "3B", "HR", "SB", "CS", "SF", "DP", "Walk"]].div(df["PA"], axis=0)

    return df, sublists

def update_names_and_ranks(df, df_now_names, df_random, coef, intercept):
    df_now_stats = pd.DataFrame()
    df_now_event = pd.DataFrame()

    df_now_stats = pd.concat([df_now_stats, df.loc[:, ["BA", "SLG", "OBP", "OPS"]]], ignore_index=True)
    df_now_event = pd.concat([df_now_event, df.loc[:, ["Hit", "2B", "3B", "HR", "SB", "CS", "SF", "DP", "Walk"]]], ignore_index=True)

    now_values = intercept + np.dot(df_now_event, coef)
    df_now_names["BA"] = df_now_stats["BA"]
    df_now_names["value"] = now_values.reshape(-1,)

    for i in df_now_names.index:
        df_random = pd.read_csv("random.csv")
        va = df_now_names.loc[i, "value"].astype(float)
        df_random.loc[len(df_random)] = va
        df_random['rank'] = df_random['value'].rank()
        rank = df_random[df_random['value'] == va]['rank'].values[0]
        score = int(rank) / 10
        df_random['rank'] = df_random['value'].rank(ascending=False)
        ranking = df_random[df_random['value'] == va]['rank'].values[0]
        ranking = int(ranking)
        df_now_names.loc[i, "rank"] = ranking
        df_now_names.loc[i, "score"] = score

    df_now_names = df_now_names.sort_values(by="value", ascending=False)
    df_now_names["value"] = df_now_names["value"].round(2)

    return df_now_names

# Streamlitのインターフェース
"""
### 得点予測値計算アプリ

&thinsp;選手の実質的な得点力を計算するよ

##### &thinsp;

##### 個人成績を評価
"""
player_stats = st.text_input("選手の個人成績から**≪試合～出塁率≫**までをコピペしてね！")

st.write("https://npb.jp/bis/players/ 検索用")

if st.button('実行'):
    # df_randomを都度読み込む
    df_random = pd.read_csv("random.csv")

    # プレイヤースタッツを処理
    df = process_player_stats(player_stats)

    # イベントスタッツを計算
    event, stats = calculate_event_stats(df)

    # 予測値を計算
    value = predict_linear_regression(event.values[0])

    # df_randomを更新
    score, ranking = update_random_df(value, df_random)

    # 結果
    result = (f"打率: {round(stats['BA'][0], 3)}&emsp;出塁率: {round(stats['OBP'][0], 3)}&emsp;長打率: {round(stats['SLG'][0], 3)}&emsp;得点予測値: {round(value, 2)}&emsp;SCORE: {score}点 ({ranking}位/1000人)")
    st.write(result)
    st.progress(score / 100)

"""
##### シーズン成績を評価
"""

years = list(range(2005, 2025))[::-1]
leagues = {'セリーグ': 'c', 'パリーグ': 'p'}

col1, col2 = st.columns(2)
# 年度の選択
with col1:
    selected_year = st.selectbox('年度を選択してね！', years)

# リーグの選択（右側）
with col2:
    selected_league = st.selectbox('リーグを選択してね！', list(leagues.keys()))

# URLの構築
format_code = leagues[selected_league]
url = f"https://npb.jp/bis/{selected_year}/stats/bat_{format_code}.html"

if st.button(' 実行'):
    df_random = pd.read_csv("random.csv")

    # URLからデータを取得
    df, sublists = process_url_data(url)

    # データフレームの初期化
    df_now_names = []
    data = []

    for i in range(len(sublists)):
        data.append([sublists[i][1]])
        df_now_names = pd.DataFrame(data, columns=['NAME'])
        delete = df_now_names["NAME"].str.contains("打撃成績")
        df_now_names.drop(df_now_names[delete].index, inplace=True)
        df_now_names["NAME"].str.contains(">> パ・リーグ 打撃成績")
        df_now_names["NAME"].str.contains(">> セ・リーグ 打撃成績")

    # 現在の統計とランダムデータフレームの更新
    df = update_names_and_ranks(df, df_now_names, df_random, coef, intercept)
    df.index = df.index + 1
    df.columns = ['選手名', '打率', '得点予測値','RANK','SCORE'] 
    df.index.name = '打率順位'
    df['打率'] = df['打率'].map('{:.3f}'.format)
    df['得点予測値'] = df['得点予測値'].map('{:.2f}'.format)
    df['SCORE'] = df['SCORE'].map('{:.1f}'.format)
    df['RANK'] = df['RANK'].map('{:.0f}'.format)
  
    st.dataframe(df, width=500, height=800, use_container_width=False, hide_index=False)
