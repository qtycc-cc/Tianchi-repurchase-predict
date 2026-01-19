import gc
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from util import calculate_group_mi
from model.iwt_classifier import IWT_Classifier

def process_data():
    data_user_info = pd.read_csv("./data/user_info_format1.csv")
    data_train = pd.read_csv("./data/train_format1.csv")
    data_test = pd.read_csv("./data/test_format1.csv")
    d_types = {'user_id': 'int32', 'item_id': 'int32', 'cat_id': 'int16', 'seller_id': 'int16', 'brand_id': 'float32',
               'time_stamp': 'int16', 'action_type': 'int8'}
    data_user_log = pd.read_csv("./data/user_log_format1.csv", dtype=d_types)

    train_main, train_ratio = train_test_split(
        data_train,
        test_size=2/9,
        random_state=42,
        stratify=data_train['label']
    )
    data_train = train_main

    merchant_label_ratio = train_ratio.groupby('merchant_id')['label'] \
        .mean().reset_index() \
        .rename(columns={'label': 'merchant_label_ratio'})

    del train_ratio
    gc.collect()

    data_train["origin"] = "train"
    data_test["origin"] = "test"
    data = pd.concat([data_train, data_test], sort=False)
    data = data.drop(["prob"], axis=1)
    del data_train, data_test
    gc.collect()

    list = [data, data_user_log, data_user_info]
    for df in list:
        fcols = df.select_dtypes('float').columns
        icols = df.select_dtypes('integer').columns
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
        df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    data_user_log.rename(columns={"seller_id": "merchant_id"}, inplace=True)

    data_user_info["age_range"].fillna(0, inplace=True)
    data_user_info["gender"].fillna(2, inplace=True)
    data_user_log["brand_id"].fillna(0, inplace=True)

    # 按user_id分组
    groups = data_user_log.groupby(["user_id"])
    # 统计交互总次数
    temp = groups.size().reset_index().rename(columns={0: "u1"})
    for df in [data, temp]:
        df['user_id'] = df['user_id'].astype('int64')
    data = pd.merge(data, temp, on="user_id", how="left")

    # 统计交互过的商品、品类、品牌、商家数
    temp = groups[['item_id', 'cat_id', 'merchant_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'u3', 'cat_id': 'u4', 'merchant_id': 'u5', 'brand_id': 'u6'})
    data = data.merge(temp, on="user_id", how="left")

    # 统计点击、加购物车、购买、收藏的操作次数
    temp = groups['action_type'].value_counts().unstack().reset_index().rename(
        columns={0: 'u7', 1: 'u8', 2: 'u9', 3: 'u10'})
    data = data.merge(temp, on="user_id", how="left")

    # 统计购买点击比
    data["u11"] = data["u9"] / data["u7"]
    # 复购率 = 复购过的商家数/购买过的总商家数
    # 按user_id,merchant_id分组，购买天数>1则复购标记为1，反之为0
    groups_rb = data_user_log[data_user_log["action_type"] == 2].groupby(["user_id", "merchant_id"])
    temp_rb = groups_rb.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "n_days"})
    temp_rb["label_um"] = [(1 if x > 1 else 0) for x in temp_rb["n_days"]]

    # 与data进行匹配
    temp = temp_rb.groupby(["user_id", "label_um"]).size().unstack(fill_value=0).reset_index()
    temp["u12"] = temp[1] / (temp[0] + temp[1])

    data = data.merge(temp[["user_id", "u12"]], on="user_id", how="left")

    # 性别、年龄独热编码处理
    data = data.merge(data_user_info, on="user_id", how="left")

    temp = pd.get_dummies(data["age_range"], prefix="age", dtype=np.int32)
    temp2 = pd.get_dummies(data["gender"], prefix="gender", dtype=np.int32)

    data = pd.concat([data, temp, temp2], axis=1)
    data.drop(columns=["age_range", "gender"], inplace=True)

    # 按merchant_id分组
    groups = data_user_log.groupby(["merchant_id"])
    # 统计交互总次数
    temp = groups.size().reset_index().rename(columns={0: "m1"})
    data = pd.merge(data, temp, on="merchant_id", how="left")
    # 统计交互天数
    temp = groups.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "m2"})
    data = data.merge(temp, on="merchant_id", how="left")
    # 统计交互过的商品、品类、品牌、用户数
    temp = groups[['item_id', 'cat_id', 'user_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'm3', 'cat_id': 'm4', 'user_id': 'm5', 'brand_id': 'm6'})
    data = data.merge(temp, on="merchant_id", how="left")

    # 统计点击、加购物车、购买、收藏的操作次数
    temp = groups['action_type'].value_counts().unstack().reset_index().rename(
        columns={0: 'm7', 1: 'm8', 2: 'm9', 3: 'm10'})
    data = data.merge(temp, on="merchant_id", how="left")

    # 统计购买点击比
    data["m11"] = data["m9"] / data["m7"]
    # 复购率 = 复购过的用户数/购买过的总用户数
    # 按user_id,merchant_id分组，购买天数>1则复购标记为1，反之为0（在上一步已计算）
    # 与data进行匹配
    temp = temp_rb.groupby(["merchant_id", "label_um"]).size().unstack(fill_value=0).reset_index()
    temp["m12"] = temp[1] / (temp[0] + temp[1])

    data = data.merge(temp[["merchant_id", "m12"]], on="merchant_id", how="left")

    # 按user_id,merchant_id分组
    groups = data_user_log.groupby(['user_id', 'merchant_id'])
    # 统计交互总次数
    temp = groups.size().reset_index().rename(columns={0: "um1"})
    data = pd.merge(data, temp, on=["merchant_id", "user_id"], how="left")

    # 统计交互天数
    temp = groups.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "um2"})
    data = data.merge(temp, on=["merchant_id", "user_id"], how="left")

    # 统计交互过的商品、品类、品牌数
    temp = groups[['item_id', 'cat_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'um3', 'cat_id': 'um4', 'brand_id': 'um5'})
    data = data.merge(temp, on=["merchant_id", "user_id"], how="left")

    # 统计点击、加购物车、购买、收藏的操作次数
    temp = groups['action_type'].value_counts().unstack().reset_index().rename(
        columns={0: 'um6', 1: 'um7', 2: 'um8', 3: 'um9'})
    data = data.merge(temp, on=["merchant_id", "user_id"], how="left")

    # 统计购买点击比
    data["um10"] = data["um8"] / data["um6"]
    data = data.merge(merchant_label_ratio, on='merchant_id', how='left')
    del merchant_label_ratio, temp, temp2, groups, groups_rb
    gc.collect()

    data.fillna(0, inplace=True)  # !important
    data.to_csv("./data/features.csv", index=False)

def process_data_1():
    user_log = pd.read_csv('data/user_log_format1.csv')
    user_info = pd.read_csv('data/user_info_format1.csv')
    train = pd.read_csv('data/train_format1.csv')
    test = pd.read_csv('data/test_format1.csv')

    user_log.rename(columns={"seller_id": "merchant_id"}, inplace=True)

    user_info["age_range"].fillna(0, inplace=True)
    user_info["gender"].fillna(2, inplace=True)
    user_log["brand_id"].fillna(0, inplace=True)

    train_main, train_ratio = train_test_split(
        train,
        test_size=3 / 9,
        random_state=42,
        stratify=train['label']
    )

    del train
    gc.collect()

    train_main["origin"] = "train"
    test["origin"] = "test"
    data = pd.concat([train_main, test], sort=False)
    data = data.drop(["prob"], axis=1)

    del train_main, test
    gc.collect()

    merchant_label_ratio = train_ratio.groupby('merchant_id')['label'].mean().reset_index().rename(
        columns={'label': 'merchant_label_ratio'})
    data = data.merge(merchant_label_ratio, on='merchant_id', how='left')

    del train_ratio, merchant_label_ratio
    gc.collect()

    # 性别、年龄独热编码处理
    data = data.merge(user_info, on="user_id", how="left")

    temp = pd.get_dummies(data["age_range"], prefix="age")
    temp2 = pd.get_dummies(data["gender"], prefix="gender")

    data = pd.concat([data, temp, temp2], axis=1)
    data.drop(columns=["age_range", "gender"], inplace=True)

    del temp, temp2
    gc.collect()

    # 按user_id,merchant_id分组，购买天数>1则复购标记为1，反之为0
    groups_rb = user_log[user_log["action_type"] == 2].groupby(["user_id", "merchant_id"])
    temp_rb = groups_rb.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "n_days"})
    temp_rb["label_um"] = [(1 if x > 1 else 0) for x in temp_rb["n_days"]]

    groups = user_log.groupby(["user_id"])
    # 日志数
    temp = groups.size().reset_index().rename(columns={0: "count_u"})
    data = pd.merge(data, temp, on="user_id", how="left")

    # 交互天数
    temp = groups.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "days_u"})
    data = data.merge(temp, on="user_id", how="left")

    # 访问商品，品类，品牌，商家数
    temp = groups[['item_id', 'cat_id', 'merchant_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'item_count_u', 'cat_id': 'cat_count_u', 'merchant_id': 'merchant_count_u',
        'brand_id': 'brand_count_u'})
    data = data.merge(temp, on="user_id", how="left")

    # 各行为类型次数
    temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={
        0: 'view_count_u', 1: 'cart_count_u', 2: 'buy_count_u', 3: 'fav_count_u'})
    data = data.merge(temp, on="user_id", how="left")

    # 行为权重
    data['action_weight_u'] = (
                data['view_count_u'] * 0.1 + data['cart_count_u'] * 0.2 + data['fav_count_u'] * 0.2 + data[
            'buy_count_u'] * 0.5)

    # 统计购买点击比
    data["buy_view_ratio_u"] = data["buy_count_u"] / data["view_count_u"]
    # 统计购买收藏比
    data['buy_fav_ratio_u'] = data['buy_count_u'] / data['fav_count_u']
    # 统计购买加购比
    data['buy_cart_ratio_u'] = data['buy_count_u'] / data['cart_count_u']
    # 购买频率
    data['buy_freq_u'] = data['buy_count_u'] / data['count_u']

    # 复购率 = 复购过的商家数/购买过的总商家数
    temp = temp_rb.groupby(["user_id", "label_um"]).size().unstack(fill_value=0).reset_index()
    temp["repurchase_rate_u"] = temp[1] / (temp[0] + temp[1])
    data = data.merge(temp[["user_id", "repurchase_rate_u"]], on="user_id", how="left")

    # 购买量/购买商家数
    temp = user_log[user_log['action_type'] == 2].groupby(['user_id']).merchant_id.nunique().reset_index().rename(
        columns={'merchant_id': 'merchant_buy_count'})
    data = data.merge(temp, on='user_id', how='left')
    data['loyal_u'] = data['buy_count_u'] / data['merchant_buy_count']

    groups = user_log.groupby(["merchant_id"])

    # 日志数
    temp = groups.size().reset_index().rename(columns={0: "count_m"})
    data = data.merge(temp, on="merchant_id", how="left")

    # 交互天数
    temp = groups.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "days_m"})
    data = data.merge(temp, on="merchant_id", how="left")

    # 访问商品，品类，品牌，用户数
    temp = groups[['item_id', 'cat_id', 'user_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'item_count_m', 'cat_id': 'cat_count_m', 'user_id': 'user_count_m', 'brand_id': 'brand_count_m'})
    data = data.merge(temp, on="merchant_id", how="left")

    # 各行为类型次数
    temp = groups.action_type.value_counts().unstack().reset_index().rename(columns={
        0: 'view_count_m', 1: 'cart_count_m', 2: 'buy_count_m', 3: 'fav_count_m'})
    data = data.merge(temp, on="merchant_id", how="left")

    # 行为权重
    data['action_weight_m'] = (
                data['view_count_m'] * 0.1 + data['cart_count_m'] * 0.2 + data['fav_count_m'] * 0.2 + data[
            'buy_count_m'] * 0.5)

    # 统计购买点击比
    data["buy_view_ratio_m"] = data["buy_count_m"] / data["view_count_m"]
    # 统计购买收藏比
    data['buy_fav_ratio_m'] = data['buy_count_m'] / data['fav_count_m']
    # 统计购买加购比
    data['buy_cart_ratio_m'] = data['buy_count_m'] / data['cart_count_m']
    # 购买频率
    data['buy_freq_m'] = data['buy_count_m'] / data['count_m']

    # 复购率
    temp = temp_rb.groupby(["merchant_id", "label_um"]).size().unstack(fill_value=0).reset_index()
    temp["repurchase_rate_m"] = temp[1] / (temp[0] + temp[1])
    data = data.merge(temp[["merchant_id", "repurchase_rate_m"]], on="merchant_id", how="left")

    # 购买量/购买用户数
    temp = user_log[user_log['action_type'] == 2].groupby(['merchant_id']).user_id.nunique().reset_index().rename(
        columns={'user_id': 'user_buy_count'})
    data = data.merge(temp, on='merchant_id', how='left')
    data['loyal_m'] = data['buy_count_m'] / data['user_buy_count']

    groups = user_log.groupby(['user_id', 'merchant_id'])

    # 日志数
    temp = groups.size().reset_index().rename(columns={0: "count_um"})
    data = data.merge(temp, on=["user_id", "merchant_id"], how="left")

    # 交互天数
    temp = groups.time_stamp.nunique().reset_index().rename(columns={"time_stamp": "days_um"})
    data = data.merge(temp, on=["user_id", "merchant_id"], how="left")

    # 访问商品，品类，品牌数
    temp = groups[['item_id', 'cat_id', 'brand_id']].nunique().reset_index().rename(columns={
        'item_id': 'item_count_um', 'cat_id': 'cat_count_um', 'brand_id': 'brand_count_um'})
    data = data.merge(temp, on=["user_id", "merchant_id"], how="left")

    # 各行为类型次数
    temp = groups.action_type.value_counts().unstack().reset_index().rename(columns={
        0: 'view_count_um', 1: 'cart_count_um', 2: 'buy_count_um', 3: 'fav_count_um'})
    data = data.merge(temp, on=["user_id", "merchant_id"], how="left")

    # 行为权重
    data['action_weight_um'] = (
                data['view_count_um'] * 0.1 + data['cart_count_um'] * 0.2 + data['fav_count_um'] * 0.2 + data[
            'buy_count_um'] * 0.5)
    # 统计购买点击比
    data["buy_view_ratio_um"] = data["buy_count_um"] / data["view_count_um"]
    # 统计购买收藏比
    data['buy_fav_ratio_um'] = data['buy_count_um'] / data['fav_count_um']
    # 统计购买加购比
    data['buy_cart_ratio_um'] = data['buy_count_um'] / data['cart_count_um']
    # 购买频率
    data['buy_freq_um'] = data['buy_count_um'] / data['count_um']

    # 交互点击比
    data['view_ratio'] = data['view_count_um'] / data['view_count_u']
    # 交互加购比
    data['cart_ratio'] = data['cart_count_um'] / data['cart_count_u']
    # 交互收藏比
    data['fav_ratio'] = data['fav_count_um'] / data['fav_count_u']
    # 交互购买比
    data['buy_ratio'] = data['buy_count_um'] / data['buy_count_u']

    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = data[numerical_cols].fillna(0)

    gc.collect()
    data.to_csv("./data/features1.csv", index=False)

def main():
    data = pd.read_csv("./data/features1.csv")
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, Y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']

    train_x, valid_x, train_y, valid_y = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_groups = train_x.shape[1]
    num_features = train_x.shape[1]
    avg_size = round(num_features // num_groups)
    gidx_list = []
    for k in range(num_groups):
        gidx_list.extend([k] * avg_size)
    gidx_list.extend([num_groups - 1] * (num_features - avg_size * num_groups))
    gidx = torch.tensor(gidx_list, dtype=torch.long, device=device)
    sgidx = []
    for kki in range(num_groups):
        idx = torch.where(gidx == kki)[0]
        sgidx.append(idx)

    # gmi = calculate_group_mi(
    #     torch.tensor(train_x.to_numpy(), dtype=torch.float32, device=device),
    #     torch.tensor(train_y.to_numpy(), dtype=torch.float32, device=device),
    #     gidx,
    #     sgidx,
    # )

    pipeline = make_pipeline(
        StandardScaler(), IWT_Classifier(
            num_groups=num_groups,
            s=32,
            gidx=gidx,
            strategy='B',
            # mu=0.7,
            # gmi=gmi,
            verbose=True,
            draw_loss=True
        )
    )

    pipeline.fit(train_x, train_y)

    auc_lr = roc_auc_score(valid_y, pipeline.predict_proba(valid_x)[:, 1])
    print(f'IWT LR roc_auc: {auc_lr}')
    # 0.6827734596499988 B 32

if __name__ == "__main__":
    # process_data_1()
    main()