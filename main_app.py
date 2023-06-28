import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split


st.title('機械学習アプリ')

uploaded_file = st.file_uploader("ファイルをアップロード")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    #dfがちゃんと読み取れているか確かめる
    #st.write(df.columns)

    st.divider()
    st.title('データの準備')

    features = st.multiselect('特徴量の選択',
                              df.columns.tolist(),#どのような選択肢
                              df.columns.tolist())#デフォルトで選択されている
    
    target = st.selectbox('ターゲットの選択',
                          df.columns.tolist())

    #テストデータを分ける
    test_size = st.slider('テストデータのサイズ', 0.0, 1.0, 0.5)

    df_train, df_test = train_test_split(df,
                                         test_size=test_size,
                                         random_state=1)

    st.divider()
    st.title('モデリング')

    model_option = st.selectbox(
        'モデルを選択',
        ['決定木', 'ランダムフォーレスト', 'ブースティング決定木']
    )

    if model_option == '決定木':
        #決定木の準備
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth=4)

    elif model_option == 'ランダムフォーレスト':
        from sklearn.ensemble  import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200,
                                       max_depth=4,
                                       random_state=1)

    elif model_option == 'ブースティング決定木':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200,
                                           max_depth=4,
                                           random_state=1)

    #確かめるために
    #st.write(model)

    button = st.button('学習開始')

    if button:
        #モデルの学習
        model.fit(df_train[features], df_train[target])

        st.divider()
        st.title('評価')

        pred_train = model.predict(df_train[features]) #学習データの予測
        pred_test = model.predict(df_test[features]) #テストデータの予測

        st.write('F1スコア(0~1:1であればあるほど良い)')
        from sklearn.metrics import f1_score
        st.write('学習データ:', f1_score(df_train[target], pred_train))
        st.write('テストデータ:', f1_score(df_test[target], pred_test))





