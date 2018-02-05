import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion

import time
import os
import sys
from sys import argv
import pickle

import lightgbm as lgb


def create_and_evaluate_model(param, n_lgbm_iter=100):
    
    param['metric'] = ['auc', 'binary_logloss']
    param['objective'] = 'binary'
    param['verbosity'] = -1
    
    train_data = lgb.Dataset(X_train, label=y_train)
    lgbm = lgb.train(param, train_data, n_lgbm_iter)

    return lgbm


dataset_name = argv[1]
optimal_params_filename = argv[2]
results_dir = argv[3]

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
lgbm = create_and_evaluate_model(best_params)

# get predictions for test set
preds = cls.predict(X_test)
dt_preds = pd.DataFrame({"preds": preds, "actual": y_test,
                         "prefix_nr": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"]})

# write AUC for every prefix length
with open(os.path.join(results_dir, "results_%s.csv" % dataset_name), 'w') as fout:
    fout.write("dataset;nr_events;auc\n")

    for i in range(min_prefix_length, max_prefix_length+1):
        tmp = dt_preds[dt_preds.prefix_nr==i]
        if len(tmp.actual.unique()) > 1:
            auc = roc_auc_score(tmp.actual, tmp.preds)
            fout.write("%s;%s;%s\n" % (dataset_name, i, auc))
            
print(time.time() - start)
            