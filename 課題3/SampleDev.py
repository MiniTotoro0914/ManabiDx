import os 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import seaborn

## Google Colabを利用する場合
path = 'C:/local_persnal_dev/ManabiDx/課題3/'

## ローカルの場合
# path = "./"

# 想定している場所にファイルがあるかどうかの確認
# 出力想定は下記の通り（順不同）
# ['PBL05_sample_code.ipynb','sample_submission.csv','category_names.csv','test.csv','sales_history.csv','item_categories.csv']
os.listdir(path)


sales_history = pd.read_csv(path+ 'data/sales_history.csv')
category_names = pd.read_csv(path+ 'data/category_names.csv')
item_categories = pd.read_csv(path+ 'data/item_categories.csv')
test_data = pd.read_csv(path+ 'data/test.csv', index_col=0)
sub = pd.read_csv(path + 'data/sample_submission.csv', header=None)

##print(sales_history.info())
##print(category_names.info())
##print(item_categories.info())
##print(test_data.info())

# 「日付」カラムの文字列から、「年」「月」の情報を抽出する
# lambda 引数: 式)
sales_history['年'] = sales_history['日付'].apply(lambda x: x.split('-')[0])
sales_history['月'] = sales_history['日付'].apply(lambda x: x.split('-')[1])

##print(sales_history.head())
sales_month = sales_history.groupby(['年','月','店舗ID','商品ID']).agg({'売上個数': 'sum'}).reset_index()
##print(sales_month)
##print(sales_history.groupby(['月','日付','店舗ID','商品ID']).sum())

# 売上(sale)データと商品カテゴリID(cats)データの統合
sales_month = pd.merge(sales_month, item_categories , on='商品ID', how='left')
##sales_month = pd.merge(sales_month, category_names , on='商品カテゴリID', how='left')

print(sales_month.head())
print(sales_month.info())

## 日付項目の数値化
sales_month['年'] = sales_month['年'].astype('int')
sales_month['月'] = sales_month['月'].astype('int')

print(sales_month.info())

test_data['年'] = 2019
test_data['月'] = 12 

test_data = pd.merge(test_data, item_categories , on='商品ID', how='left')
print(test_data.info())

# 説明変数に該当するカラムの一覧
feature_columns = ['商品ID', '店舗ID', '商品カテゴリID', '年', '月']

# 学習用データの整理
X_train = sales_month[feature_columns] # 学習用データの説明変数
y_train = sales_month['売上個数'] # 学習用データの目的変数

# テスト用データの整理
X_test = test_data[feature_columns] # テスト用データの説明変数

##print(X_train.head())
##print(X_test.head())
print(y_train.head())

# モデルの型の生成
model = DecisionTreeRegressor()

# モデルの学習
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)


# sample_submissionの右端のカラムに予測値を代入する。
sub[sub.columns[-1]] = y_pred

# 提出ファイルの生成
sub.to_csv('my_submission.csv', index=False, header=False)
