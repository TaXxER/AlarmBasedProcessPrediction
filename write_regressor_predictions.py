import EncoderFactory
from DatasetManager import DatasetManager
from calibration_wrappers import LGBMCalibrationWrapper

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.calibration import CalibratedClassifierCV

import time
import os
import sys
from sys import argv
import pickle

import lightgbm as lgb


def create_regression_model(X_train, y_train, param, n_lgbm_iter=100):
    
    param['metric'] = ['mae']
    param['objective'] = 'regression'
    param['verbosity'] = -1
    
    train_data = lgb.Dataset(X_train, label=y_train)
    lgbm = lgb.train(param, train_data, n_lgbm_iter)
    
    return lgbm
    
    
dataset_name = argv[1]
optimal_params_filename = argv[2]
preds_dir = argv[3]
results_dir = argv[4]
calibrate = bool(argv[5])

train_ratio = 0.8

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))
    
print('Preparing data...')
start = time.time()

# read the data
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()

min_prefix_length = 1
if "bpic2017" in dataset_name:
    max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.95))
else:
    max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.95))

cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols, 
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                    'fillna': True}
    
# split into training and test
train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

# generate data where each prefix is a separate instance
dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

# encode all prefixes
feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in ["static", "agg"]])
X_train = feature_combiner.fit_transform(dt_train_prefixes)
X_test = feature_combiner.fit_transform(dt_test_prefixes)
y_train = dataset_manager.get_label_numeric(dt_train_prefixes)
y_test = dataset_manager.get_label_numeric(dt_test_prefixes)

# train the model with pre-tuned parameters
with open(optimal_params_filename, "rb") as fin:
    best_params = pickle.load(fin)

dt_preds = pd.DataFrame({"prefix_nr": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"],
                         "case_id": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})

# get predictions for each dataset
X_train = pd.DataFrame(X_train, index=dt_train_prefixes.groupby(dataset_manager.case_id_col).first().reset_index()[dataset_manager.case_id_col])
X_test = pd.DataFrame(X_test, index=dt_test_prefixes.groupby(dataset_manager.case_id_col).first().reset_index()[dataset_manager.case_id_col])
    
# predict remaining case length
X_train["prefix_nr"] = dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"]
X_train["case_length"] = dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["case_length"]
X_train["orig_case_id"] = dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]
X_train["remaining_case_length"] = X_train["case_length"] - X_train["prefix_nr"]
gbm_case_length = create_regression_model(X_train=X_train.drop(["remaining_case_length", "case_length", "prefix_nr", 
                                                                "orig_case_id"], axis=1),
                                          y_train=X_train["remaining_case_length"],
                                          param=best_params)
preds_case_length = gbm_case_length.predict(X_test)
dt_preds["predicted_case_length"] = preds_case_length
    
# predict conf delta
dt_preds_cls_train = pd.read_csv(os.path.join(preds_dir, "preds_train_%s.csv" % dataset_name), sep=";").drop("actual", axis=1)
dt_preds_cls_train["case_id"] = dt_preds_cls_train["case_id"].astype(str)
X_train = X_train.merge(dt_preds_cls_train, left_on=["orig_case_id", "prefix_nr"], right_on=["case_id", "prefix_nr"]).drop("case_id", axis=1)

X_train["predicted_proba_next"] = X_train.sort_values("prefix_nr", ascending=True).groupby("orig_case_id")["predicted_proba"].shift(-1)
X_train["conf_current"] = abs(0.5 - X_train["predicted_proba"]) + 0.5
X_train["conf_current_next"] = abs(0.5 - X_train["predicted_proba_next"]) + 0.5
X_train["conf_diff"] = X_train["conf_current_next"] - X_train["conf_current"]
    
X_train = X_train.dropna()
    
gbm_conf_delta = create_regression_model(X_train=X_train.drop(["remaining_case_length", "case_length", "prefix_nr",
                                                               "orig_case_id", "predicted_proba", "predicted_proba_next",
                                                               "conf_current", "conf_current_next"], axis=1),
                                         y_train=X_train["conf_diff"],
                                         param=best_params)
preds_conf_delta = gbm_conf_delta.predict(X_test)
dt_preds["predicted_conf_delta"] = preds_conf_delta
    
dt_preds.to_csv(os.path.join(results_dir, "preds_%s.csv" % dataset_name), sep=";", index=False)
   
print(time.time() - start)
            