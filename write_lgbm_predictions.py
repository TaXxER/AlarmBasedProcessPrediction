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


def create_model(param, n_lgbm_iter=100, calibrate=False):
    
    param['metric'] = ['auc', 'binary_logloss']
    param['objective'] = 'binary'
    param['verbosity'] = -1
    
    train_data = lgb.Dataset(X_train, label=y_train)
    lgbm = lgb.train(param, train_data, n_lgbm_iter)
    
    if calibrate:
        wrapper = LGBMCalibrationWrapper(lgbm)
        cls = CalibratedClassifierCV(wrapper, cv="prefit", method='sigmoid')
        cls.fit(X_val, y_val)
        return cls
    else:
        return lgbm


dataset_name = argv[1]
optimal_params_filename = argv[2]
results_dir = argv[3]
calibrate = bool(argv[4])

train_ratio = 0.8
val_ratio = 0.2

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

if calibrate:
    train, val = dataset_manager.split_val(train, val_ratio)
    dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)
    
# generate data where each prefix is a separate instance
dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

# encode all prefixes
feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in ["static", "agg"]])
X_train = feature_combiner.fit_transform(dt_train_prefixes)
X_test = feature_combiner.fit_transform(dt_test_prefixes)
y_train = dataset_manager.get_label_numeric(dt_train_prefixes)
y_test = dataset_manager.get_label_numeric(dt_test_prefixes)

if calibrate:
    X_val = feature_combiner.fit_transform(dt_val_prefixes)
    y_val = dataset_manager.get_label_numeric(dt_val_prefixes)

# train the model with pre-tuned parameters
with open(optimal_params_filename, "rb") as fin:
    best_params = pickle.load(fin)
gbm = create_model(best_params, calibrate=calibrate)

# get predictions for test set
if calibrate:
    preds = gbm.predict_proba(X_test)
else:
    preds = lgbm.predict(X_test)

dt_preds = pd.DataFrame({"predicted_proba": preds[:,1], "actual": y_test,
                         "prefix_nr": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"],
                         "case_id": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})

dt_preds.to_csv(os.path.join(results_dir, "preds_%s.csv" % dataset_name), sep=";", index=False)

"""
# write AUC for every prefix length
with open(os.path.join(results_dir, "results_%s.csv" % dataset_name), 'w') as fout:
    fout.write("dataset;nr_events;auc\n")

    for i in range(min_prefix_length, max_prefix_length+1):
        tmp = dt_preds[dt_preds.prefix_nr==i]
        if len(tmp.actual.unique()) > 1:
            auc = roc_auc_score(tmp.actual, tmp.predicted_proba)
            fout.write("%s;%s;%s\n" % (dataset_name, i, auc))
"""            
print(time.time() - start)
            