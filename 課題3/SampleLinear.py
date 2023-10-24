import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from sklearn.linear_model import LinearRegression

## Google Colabを利用する場合
path = 'C:/local_persnal_dev/ManabiDx/課題3/'
os.listdir(path)

sales_history = pd.read_csv(path+ 'data/sales_history.csv')
category_names = pd.read_csv(path+ 'data/category_names.csv')
item_categories = pd.read_csv(path+ 'data/item_categories.csv')
test_data = pd.read_csv(path+ 'data/test.csv')
sub = pd.read_csv(path + 'data/sample_submission.csv', header=None)

# 「日付」カラムの文字列から、「年」「月」の情報を抽出する
# lambda 引数: 式)
sales_history['年'] = sales_history['日付'].apply(lambda x: x.split('-')[0])
sales_history['月'] = sales_history['日付'].apply(lambda x: x.split('-')[1])

##print(sales_history.head())
sales_month = sales_history.groupby(['年','月','店舗ID','商品ID']).agg({'売上個数': 'sum'}).reset_index()

# 売上(sale)データと商品カテゴリID(cats)データの統合
sales_month = pd.merge(sales_month, item_categories , on='商品ID', how='left')
test_data = pd.merge(test_data, item_categories , on='商品ID', how='left')

## 日付項目の数値化
sales_month['年'] = sales_month['年'].astype('int')
sales_month['月'] = sales_month['月'].astype('int')
test_data['年'] = 2019
test_data['月'] = 12 

# 説明変数に該当するカラムの一覧
feature_columns = ['商品ID','商品カテゴリID', '店舗ID', '年', '月']

# 学習用データの整理
X_train = sales_month[feature_columns] # 学習用データの説明変数
y_train = sales_month['売上個数'] # 学習用データの目的変数

print(test_data.info())

# テスト用データの整理
X_test = test_data[feature_columns] # テスト用データの説明変数

lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)

# sample_submissionの右端のカラムに予測値を代入する。
sub[sub.columns[-1]] = y_pred

# 提出ファイルの生成
sub.to_csv('my_submission_linear_reg2.csv', index=False, header=False)

