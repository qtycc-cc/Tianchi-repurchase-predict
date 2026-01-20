import gc
import torch
import numpy as np
import pandas as pd
from typing import Literal
from pytabkit import TabM_D_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from util import calculate_group_mi, result_beep
from model.iwt_classifier import IWT_Classifier

def process_data(expose_size: float):
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
        test_size=expose_size,
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

    temp = pd.get_dummies(data["age_range"], prefix="age", dtype='int')
    temp2 = pd.get_dummies(data["gender"], prefix="gender", dtype='int')

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
    data.to_csv("./data/features.csv", index=False)

def get_best_score():
    data = pd.read_csv(f'./data/features.csv')
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

    gmi = calculate_group_mi(
        torch.tensor(train_x.to_numpy(), dtype=torch.float32, device=device),
        torch.tensor(train_y.to_numpy(), dtype=torch.float32, device=device),
        gidx,
        sgidx,
    )

    print(f'Gmi = {gmi}')

    res = []
    for strategy in ['B', 'M']:
        for s in range(1, num_groups + 1):
            for mu in [0.6, 0.65, 0.7, 0.75]:
                pipeline = make_pipeline(
                    StandardScaler(), IWT_Classifier(
                        num_groups=num_groups,
                        s=s,
                        gidx=gidx,
                        strategy=strategy,
                        mu=mu,
                        gmi=gmi,
                        # verbose=True,
                        # draw_loss=True
                    )
                )
                pipeline.fit(train_x, train_y)

                auc_lr = roc_auc_score(valid_y, pipeline.predict_proba(valid_x)[:, 1])
                print(f"s = {s} | strategy = {strategy} | auc_lr = {auc_lr}")
                res.append({"s": s, "strategy": strategy, "mu": mu, "auc_lr": auc_lr})

    max_auc_item = max(res, key=lambda x: x["auc_lr"])

    print(f"Max AUC is: {max_auc_item['auc_lr']}")
    print(f"Best s is: {max_auc_item['s']}")
    print(f"Best strategy is: {max_auc_item['strategy']}")
    print(f"Best mu is: {max_auc_item['mu']}")

    return max_auc_item
    # 3/9
    # 0.6838649522004095 44 M 0.65
    # 0.6839076979581018 40 B

def train_with_IWT(strategy: Literal['B', 'M', 'T', 'H'], s: int, mu: float):
    data = pd.read_csv(f'./data/features.csv')
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

    if strategy == 'M':
        gmi = calculate_group_mi(
            torch.tensor(train_x.to_numpy(), dtype=torch.float32, device=device),
            torch.tensor(train_y.to_numpy(), dtype=torch.float32, device=device),
            gidx,
            sgidx,
        )
        print(f'Gmi = {gmi}')
    else:
        gmi = None

    pipeline = make_pipeline(
        StandardScaler(), IWT_Classifier(
            num_groups=num_groups,
            s=s,
            gidx=gidx,
            strategy=strategy,
            mu=mu,
            gmi=gmi,
            # verbose=True,
            # draw_loss=True
        )
    )

    pipeline.fit(train_x, train_y)

    auc_lr = roc_auc_score(valid_y, pipeline.predict_proba(valid_x)[:, 1])
    print(f'IWT LR roc_auc: {auc_lr}')

    # pipeline.fit(X, Y) 会降低分数

    support_w = pipeline.named_steps['iwt_classifier'].w_
    print(f'W = {support_w}')

    mask = support_w == 0
    non_zeros = X.iloc[:, mask]
    non_zeros.to_csv(f'data/non_zeros.csv', index=False)
    print('non_zeros.csv saved')
    del non_zeros
    gc.collect()

def train_with_tabM(use_less_feature: bool = False):
    data = pd.read_csv(f'./data/features.csv')
    train = data[data["origin"] == "train"].drop(["origin"], axis=1)
    test = data[data["origin"] == "test"].drop(["origin", "label"], axis=1)

    X, Y = train.drop(['user_id', 'merchant_id', 'label'], axis=1), train['label']
    X_test = test.drop(columns=['user_id', 'merchant_id'])

    if use_less_feature:
        non_zeros = pd.read_csv(f'./data/non_zeros.csv')
        columns_to_keep = [col for col in non_zeros.columns if col in X.columns]
        X = X[columns_to_keep]
        X_test = X_test[columns_to_keep]

    train_x, valid_x, train_y, valid_y = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    pipeline = make_pipeline(TabM_D_Classifier(verbosity=2, val_metric_name='1-auc_ovr', allow_amp=True))

    pipeline.fit(train_x, train_y)

    auc_lr = roc_auc_score(valid_y, pipeline.predict_proba(valid_x)[:, 1])

    print(f'TabM roc_auc: {auc_lr}')

    pipeline.fit(X, Y)

    test_pred = pipeline.predict_proba(X_test)[:, 1]
    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = test_pred
    submission.to_csv('data/submission_tabM.csv', index=False)
    print('Submission csv saved!')


@result_beep
def main(use_less_feature: bool = False, is_data_saved: bool = False, is_important_features_saved: bool = False, expose_size: float = 3/9):
    if not is_data_saved:
        process_data(expose_size)
    if use_less_feature and not is_important_features_saved:
        max_auc_item = get_best_score()
        print(max_auc_item)
        train_with_IWT(strategy=max_auc_item['strategy'], s=max_auc_item['s'], mu=max_auc_item['mu'])
    train_with_tabM(use_less_feature=use_less_feature)

if __name__ == "__main__":
    # {'s': 40, 'strategy': 'B', 'mu': 0.6, 'auc_lr': 0.6839076979581018} 3/9
    # {'s': 42, 'strategy': 'B', 'mu': 0.6, 'auc_lr': 0.6790662662970054} 4/9
    main(use_less_feature=True, is_data_saved=True, is_important_features_saved=False, expose_size=4/9)