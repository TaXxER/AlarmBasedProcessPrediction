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

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt


def calculate_cost(x, costs):
    return costs[int(x['prediction']), int(x['actual'])](x)

def evaluate_model_cost(args):
    conf_threshold = args['conf_threshold']
    total_cost = 0
    for dt_preds in preds_folds:
        # trigger alarms according to conf_threshold
        dt_final = pd.DataFrame()
        unprocessed_case_ids = set(dt_preds.case_id.unique())
        for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
            tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
            tmp = tmp[tmp.predicted_proba >= conf_threshold]
            tmp["prediction"] = 1
            dt_final = pd.concat([dt_final, tmp], axis=0)
            unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == 1)]
        tmp["prediction"] = 0
        dt_final = pd.concat([dt_final, tmp], axis=0)

        case_lengths = dt_preds.groupby("case_id").prefix_nr.max().reset_index()
        case_lengths.columns = ["case_id", "case_length"]
        dt_final = dt_final.merge(case_lengths)
        
        total_cost += dt_final.apply(calculate_cost, costs=costs, axis=1).sum()

    return {'loss': total_cost, 'status': STATUS_OK, 'model': lgbm}


print('Preparing data...')
start = time.time()

dataset_name = argv[1]
optimal_params_filename = argv[2]
params_dir = argv[3]
calibrate = bool(argv[4])

train_ratio = 0.8
calibration_val_ratio = 0.2
n_splits = 3

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
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

# load pre-tuned lgbm params
with open(optimal_params_filename, "rb") as fin:
    param = pickle.load(fin)
param['metric'] = ['auc', 'binary_logloss']
param['objective'] = 'binary'
param['verbosity'] = -1
    
# split into training and test
train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    
print('Training classifiers and obtaining predictions for each fold...')
preds_folds = []
# train classifiers and save predictions for each fold
for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):

    if calibrate:
        # data for calibration
        train_chunk, val_chunk = dataset_manager.split_val(train_chunk, calibration_val_ratio)
        dt_val_prefixes = dataset_manager.generate_prefix_data(val_chunk, min_prefix_length, max_prefix_length)

    # create prefix logs
    dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, min_prefix_length, max_prefix_length)
    dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length)
    
    # encode all prefixes
    feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in ["static", "agg"]])
    X_train = feature_combiner.fit_transform(dt_train_prefixes)
    y_train = np.array(dataset_manager.get_label_numeric(dt_train_prefixes))
    if calibrate:
        X_val = feature_combiner.fit_transform(dt_val_prefixes)
        y_val = np.array(dataset_manager.get_label_numeric(dt_val_prefixes))
    X_test = feature_combiner.fit_transform(dt_test_prefixes)
    y_test = np.array(dataset_manager.get_label_numeric(dt_test_prefixes))

    train_data = lgb.Dataset(X_train, label=y_train)
    lgbm = lgb.train(param, train_data, 100)
    
    if calibrate:
        wrapper = LGBMCalibrationWrapper(lgbm)
        cls = CalibratedClassifierCV(wrapper, cv="prefit", method='sigmoid')
        cls.fit(X_val, y_val)
        preds = cls.predict_proba(X_test)[:,1]
    else:
        preds = lgbm.predict(X_test)
        
    dt_preds = pd.DataFrame({"predicted_proba": preds, "actual": y_test,
                             "prefix_nr": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"],
                             "case_id": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})
    preds_folds.append(dt_preds)


print('Optimizing parameters...')
cost_weights = [(1,1,-5), (5,1,-1), (1,5,-1), (5,1,-5), (1,5,-5), (5,5,-1), (1,1,-1)]
for c01, c10, c11 in cost_weights:
    # cost matrix
    costs = np.matrix([[lambda x: 0,
                        lambda x: c01],
                       [lambda x: c10,
                        lambda x: c11*(x['case_length'] - x['prefix_nr'])/x['case_length']]])
    
    space = {'conf_threshold': hp.uniform("conf_threshold", 0, 1)}
    trials = Trials()
    best = fmin(evaluate_model_cost, space, algo=tpe.suggest, max_evals=50, trials=trials)

    best_params = hyperopt.space_eval(space, best)

    outfile = os.path.join(params_dir, "optimal_confs_%s_%s_%s_%s.pickle" % (dataset_name, c01, c10, c11))
    # write to file
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)
