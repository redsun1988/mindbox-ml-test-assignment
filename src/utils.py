from pathlib import Path
from typing import Iterable, List, Tuple

from src.load import DataProvider
from os import environ
from scipy import sparse
import implicit
import numpy as np
from ml_metrics import mapk
import pandas as pd


def transform_to_item_user_csr_matrix(purchases: pd.DataFrame) -> sparse.csr_matrix:
    item_users = sparse.coo_matrix(
        (
            np.ones(purchases.customer_id.size, dtype=np.float32),
            (
                purchases.product_id,
                purchases.customer_id,
            )
        )
    ).tocsr()
    return item_users

def transform_to_user_sex_csr_matrix(customers: pd.DataFrame) -> sparse.csr_matrix:
    def getSexLabel(x):
        if x == "Female":
            return 1
        if x == "Male":
            return 2
        return 3
    
    user_sex = sparse.coo_matrix(
        (
            np.array([getSexLabel(s) for s in customers.sex]),
            (
                np.array([0] * customers.customer_id.size),
                np.array(customers.customer_id),
            )
        )
    ).tocsr()

    return user_sex


def get_model() -> implicit.als.AlternatingLeastSquares:
    # disable internal multithreading to speed up implicit.als.AlternatingLeastSquares.fit()
    environ['MKL_NUM_THREADS'] = '1'
    model = implicit.als.AlternatingLeastSquares(factors=20, iterations=7, regularization=100.0)
    return model


def get_recommendations(model: implicit.als.AlternatingLeastSquares, user_ids: Iterable[int], item_users: sparse.csr_matrix) -> List[List[int]]:
    user_items = item_users.T.tocsr()
    recommendations = []
    for user_id in user_ids:
        recommendations.append([x[0] for x in model.recommend(userid=user_id, user_items=user_items, N=10)])
    return recommendations


def get_purchases_by_customer(purchases: pd.DataFrame) -> Tuple[List[int], List[List[int]]]:
    relevant = purchases.groupby('customer_id')['product_id'].apply(lambda s: s.values.tolist()).reset_index()
    relevant.rename(columns={'product_id': 'product_ids'}, inplace=True)
    return relevant['customer_id'].tolist(), relevant['product_ids'].tolist()