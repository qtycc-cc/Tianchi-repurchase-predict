import gc
import torch
import numpy as np
import pandas as pd
from typing import Literal
from pytabkit import TabM_D_Classifier, TabM_HPO_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from util import calculate_group_mi, result_beep
from model.iwt_classifier import IWT_Classifier

def process_data(expose_size: float = 1/2):
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

    item_label_ratio_max = pd.merge(user_log[(user_log['action_type'] == 2)], train_ratio,
                                    how='inner').drop_duplicates()
    item_label_ratio_mean = item_label_ratio_max.groupby(['item_id'])['label'].mean().reset_index().rename(
        columns={'label': 'item_label_ratio'})
    item_label_ratio_max = pd.merge(item_label_ratio_max, item_label_ratio_mean, how='left')
    del item_label_ratio_mean
    gc.collect()
    item_label_ratio_max = item_label_ratio_max.groupby(['merchant_id'])['item_label_ratio'].max().reset_index()

    cat_label_ratio_max = pd.merge(user_log[(user_log['action_type'] == 2)], train_ratio, how='inner').drop_duplicates()
    cat_label_ratio_mean = cat_label_ratio_max.groupby(['cat_id'])['label'].mean().reset_index().rename(
        columns={'label': 'cat_label_ratio'})
    cat_label_ratio_max = pd.merge(cat_label_ratio_max, cat_label_ratio_mean, how='inner')
    del cat_label_ratio_mean
    gc.collect()
    cat_label_ratio_max = cat_label_ratio_max.groupby(['merchant_id'])['cat_label_ratio'].max().reset_index()

    brand_label_ratio_max = pd.merge(user_log[(user_log['action_type'] == 2)], train_ratio,
                                     how='inner').drop_duplicates().dropna()
    brand_label_ratio_mean = brand_label_ratio_max.groupby(['brand_id'])['label'].mean().reset_index().rename(
        columns={'label': 'brand_label_ratio'})
    brand_label_ratio_max = pd.merge(brand_label_ratio_max, brand_label_ratio_mean, how='left')
    del brand_label_ratio_mean
    gc.collect()
    brand_label_ratio_max = brand_label_ratio_max.groupby(['merchant_id'])['brand_label_ratio'].max().reset_index()

    data = data.merge(item_label_ratio_max, on='merchant_id', how='left')
    data = data.merge(cat_label_ratio_max, on='merchant_id', how='left')
    data = data.merge(brand_label_ratio_max, on='merchant_id', how='left')

    del train_ratio, merchant_label_ratio, item_label_ratio_max, cat_label_ratio_max, brand_label_ratio_max
    gc.collect()

    # 性别、年龄独热编码处理
    data = data.merge(user_info, on="user_id", how="left")

    temp = pd.get_dummies(data["age_range"], prefix="age", dtype='int32')
    temp2 = pd.get_dummies(data["gender"], prefix="gender", dtype='int32')

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

def get_best_score(equalsize: bool = True):
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

    if equalsize:
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
    else:
        feature_names = X.columns.tolist()

        num_groups = train_x.shape[1]
        num_features = train_x.shape[1]
        # 先找出性别和年龄特征的索引
        age_indices = [i for i, name in enumerate(feature_names) if name.startswith('age_')]
        gender_indices = [i for i, name in enumerate(feature_names) if name.startswith('gender_')]

        # 剩余的特征索引（除去性别和年龄）
        remaining_indices = [i for i in range(num_features) if i not in age_indices + gender_indices]

        # 现在重新计算分组
        # 性别为1组，年龄为1组，剩余的特征按平均分组
        num_remaining_groups = num_groups - 2  # 减去性别和年龄两个特殊分组

        # 对剩余的特征进行平均分组
        num_remaining_features = len(remaining_indices)
        avg_size = num_remaining_features // num_remaining_groups if num_remaining_groups > 0 else 0

        # 创建分组索引列表
        gidx_list = [0] * num_features

        # 1. 将年龄特征设为第0组
        for idx in age_indices:
            gidx_list[idx] = 0

        # 2. 将性别特征设为第1组
        for idx in gender_indices:
            gidx_list[idx] = 1

        # 3. 将剩余特征平均分配到其他组
        group_counter = 2  # 从第2组开始
        current_size = 0

        for idx in remaining_indices:
            gidx_list[idx] = group_counter
            current_size += 1

            # 如果当前组已满，且不是最后一组，则换到下一组
            if current_size >= avg_size and group_counter < num_groups - 1:
                group_counter += 1
                current_size = 0

        # 创建分组索引张量
        gidx = torch.tensor(gidx_list, dtype=torch.long, device=device)

        # 创建子组索引列表
        sgidx = []
        unique_groups = sorted(set(gidx_list))
        for group_id in unique_groups:
            idx = torch.where(gidx == group_id)[0]
            sgidx.append(idx)

    gmi = calculate_group_mi(
        torch.tensor(train_x.to_numpy(), dtype=torch.float32, device=device),
        torch.tensor(train_y.to_numpy(), dtype=torch.float32, device=device),
        gidx,
        sgidx,
        equalsize
    )

    print(f'Gmi = {gmi}')

    res = []
    for strategy in ['B', 'M']:
        for s in range(10, num_groups + 1):
            cnt = 0
            for mu in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00]:
                cnt = cnt + 1
                pipeline = make_pipeline(
                    StandardScaler(), IWT_Classifier(
                        num_groups=len(sgidx),
                        s=s,
                        gidx=gidx,
                        sgidx=sgidx,
                        strategy=strategy,
                        mu=mu,
                        gmi=gmi,
                        equalsize=equalsize,
                        # verbose=True,
                        # draw_loss=True
                    )
                )
                pipeline.fit(train_x, train_y)

                auc_lr = roc_auc_score(valid_y, pipeline.predict_proba(valid_x)[:, 1])
                print(f"s = {s} | strategy = {strategy} | auc_lr = {auc_lr}")
                res.append({"s": s, "strategy": strategy, "mu": mu, "auc_lr": auc_lr})

                if strategy == 'B' and cnt == 1: # B 策略提前终止
                    break

    max_auc_item = max(res, key=lambda x: x["auc_lr"])

    print(f"Max AUC is: {max_auc_item['auc_lr']}")
    print(f"Best s is: {max_auc_item['s']}")
    print(f"Best strategy is: {max_auc_item['strategy']}")
    print(f"Best mu is: {max_auc_item['mu']}")

    return max_auc_item

def train_with_IWT(strategy: Literal['B', 'M', 'T', 'H'], s: int, mu: float, equalsize: bool = True):
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

    if equalsize:
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
    else:
        feature_names = X.columns.tolist()

        num_groups = train_x.shape[1]
        num_features = train_x.shape[1]
        # 先找出性别和年龄特征的索引
        age_indices = [i for i, name in enumerate(feature_names) if name.startswith('age_')]
        gender_indices = [i for i, name in enumerate(feature_names) if name.startswith('gender_')]

        # 剩余的特征索引（除去性别和年龄）
        remaining_indices = [i for i in range(num_features) if i not in age_indices + gender_indices]

        # 现在重新计算分组
        # 性别为1组，年龄为1组，剩余的特征按平均分组
        num_remaining_groups = num_groups - 2  # 减去性别和年龄两个特殊分组

        # 对剩余的特征进行平均分组
        num_remaining_features = len(remaining_indices)
        avg_size = num_remaining_features // num_remaining_groups if num_remaining_groups > 0 else 0

        # 创建分组索引列表
        gidx_list = [0] * num_features

        # 1. 将年龄特征设为第0组
        for idx in age_indices:
            gidx_list[idx] = 0

        # 2. 将性别特征设为第1组
        for idx in gender_indices:
            gidx_list[idx] = 1

        # 3. 将剩余特征平均分配到其他组
        group_counter = 2  # 从第2组开始
        current_size = 0

        for idx in remaining_indices:
            gidx_list[idx] = group_counter
            current_size += 1

            # 如果当前组已满，且不是最后一组，则换到下一组
            if current_size >= avg_size and group_counter < num_groups - 1:
                group_counter += 1
                current_size = 0

        # 创建分组索引张量
        gidx = torch.tensor(gidx_list, dtype=torch.long, device=device)

        # 创建子组索引列表
        sgidx = []
        unique_groups = sorted(set(gidx_list))
        for group_id in unique_groups:
            idx = torch.where(gidx == group_id)[0]
            sgidx.append(idx)

    if strategy == 'M':
        gmi = calculate_group_mi(
            torch.tensor(train_x.to_numpy(), dtype=torch.float32, device=device),
            torch.tensor(train_y.to_numpy(), dtype=torch.float32, device=device),
            gidx,
            sgidx,
            equalsize
        )
        print(f'Gmi = {gmi}')
    else:
        gmi = None

    pipeline = make_pipeline(
        StandardScaler(), IWT_Classifier(
            num_groups=len(sgidx),
            s=s,
            gidx=gidx,
            sgidx=sgidx,
            strategy=strategy,
            mu=mu,
            gmi=gmi,
            equalsize=equalsize,
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

def train_with_tabM(n_cv: int, use_less_feature: bool = False, use_hpo: bool = False):
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

    if use_hpo:
        model = TabM_HPO_Classifier(verbosity=2, val_metric_name='1-auc_ovr', n_cv=n_cv, hpo_space_name='tabarena', use_caruana_ensembling=True, n_hyperopt_steps=50, tmp_folder='data/tmp')
    else:
        model = TabM_D_Classifier(verbosity=2, val_metric_name='1-auc_ovr', n_cv=n_cv, tmp_folder='data/tmp')

    # fit directly avoid wasting time
    model.fit(X, Y, cat_col_names=['age_0.0','age_1.0','age_2.0','age_3.0','age_4.0','age_5.0','age_6.0','age_7.0','age_8.0','gender_0.0','gender_1.0','gender_2.0'])

    test_pred = model.predict_proba(X_test)[:, 1]
    submission = test[['user_id', 'merchant_id']].copy()
    submission['prob'] = test_pred
    submission.to_csv('data/submission_tabM.csv', index=False)
    print('Submission csv saved!')


@result_beep
def main(
        n_cv:int,
        use_less_feature: bool = False,
        use_hpo: bool = False,
        is_data_saved: bool = False,
        is_important_features_saved: bool = False,
        expose_size: float = 1/2,
        equalsize: bool = True,
) -> None:
    """

    :param n_cv: tabM 交叉验证次数
    :param use_less_feature: tabM是否使用关键特征
    :param use_hpo: 是否使用参数搜索，注意此处用的是tabM-mini
    :param is_data_saved: 特征数据是否保存，若未保存，执行特征提取
    :param is_important_features_saved: 关键特征是否保存，若未保存，执行IWT筛选关键特征
    :param expose_size: 选取训练集多少用于计算商家覆盖率
    :param equalsize: IWT是否退化为单一稀疏而不是组稀疏
    :return:
    """
    if not is_data_saved:
        process_data(expose_size)
    if use_less_feature and not is_important_features_saved:
        train_with_IWT(strategy='M', s=52, mu=0.85, equalsize=equalsize)
        # train_with_IWT(strategy='B', s=39, mu=0.85, equalsize=equalsize)
    train_with_tabM(n_cv=n_cv, use_less_feature=use_less_feature, use_hpo=use_hpo)

@result_beep
def test():
    max_auc_item = get_best_score()
    print(max_auc_item)

if __name__ == "__main__":
    main(n_cv=8, use_less_feature=True, use_hpo=False, is_data_saved=True, is_important_features_saved=False, equalsize=True)