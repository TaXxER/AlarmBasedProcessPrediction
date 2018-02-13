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


def create_regression_model(X_train, y_train, param, n_lgbm_iter=100):
    
    param['metric'] = ['mae']
    param['objective'] = 'regression'
    param['verbosity'] = -1
    
    train_data = lgb.Dataset(X_train, label=y_train)
    lgbm = lgb.train(param, train_data, n_lgbm_iter)
    
    return lgbm

def calculate_cost(x, costs):
    return costs[int(x['prediction']), int(x['actual'])](x)

def evaluate_model_cost(args):
    total_cost = 0
    for dt_preds in preds_folds:
        # trigger alarms according to conf_threshold
        dt_final = pd.DataFrame()
        unprocessed_case_ids = set(dt_preds.case_id.unique())
        for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
            tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
            tmp["predicted_proba_next"] = tmp.predicted_proba + abs(tmp.predicted_conf_delta)
            
            expected_cost_now_alert = (tmp.predicted_proba * costs[1,1]({"prefix_nr": nr_events,
                                                                         "case_length": nr_events+tmp.predicted_case_length}) +
                                       (1-tmp.predicted_proba) * costs[1,0]({}))
            expected_cost_now_no_alert = (tmp.predicted_proba * costs[0,1]({}) +
                                          (1-tmp.predicted_proba) * costs[0,0]({}))
            expected_cost_next_alert = (tmp.predicted_proba_next * costs[1,1]({"prefix_nr": nr_events+1,
                                                                               "case_length": nr_events+tmp.predicted_case_length}) +
                                        (1-tmp.predicted_proba_next) * costs[1,0]({}))
            expected_cost_next_no_alert = (tmp.predicted_proba_next * costs[0,1]({}) +
                                          (1-tmp.predicted_proba_next) * costs[0,0]({}))
            
            # trigger alarm or not?
            if alpha_variant == "initial":
                alpha = args['alpha']
                expected_cost_next = alpha * expected_cost_next_alert + (1-alpha) * expected_cost_next_no_alert
                tmp = tmp[(expected_cost_now_alert <= expected_cost_next)]
            
            elif alpha_variant == "both_sides":
                alpha = args['alpha']
                expected_cost_now = alpha * expected_cost_now_alert + (1-alpha) * expected_cost_now_no_alert
                expected_cost_next = alpha * expected_cost_next_alert + (1-alpha) * expected_cost_next_no_alert
                tmp = tmp[(expected_cost_now <= expected_cost_next)]
                
            elif alpha_variant == "alpha_method":
                alpha = args['alpha']
                expected_cost_now = alpha * expected_cost_now_alert + (1-alpha) * expected_cost_now_no_alert
                expected_cost_next = alpha * expected_cost_next_alert + (1-alpha) * expected_cost_next_no_alert
                tmp = tmp[(expected_cost_now_alert <= expected_cost_now_no_alert) & (expected_cost_now <= expected_cost_next)]
                
            elif alpha_variant == "parameter_free":
                coef1 = args['coef1']
                coef2 = args['coef2']
                tmp = tmp[(expected_cost_now_alert <= expected_cost_now_no_alert + coef1) & (expected_cost_now_alert <= expected_cost_next_alert + coef2)]
            
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

    return {'loss': total_cost, 'status': STATUS_OK, 'model': dt_final}


print('Preparing data...')
start = time.time()

dataset_name = argv[1]
optimal_params_filename = argv[2]
params_dir = argv[3]
calibrate = bool(argv[4])
alpha_variant = argv[5]
cost_step = int(argv[6])

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
preds_delta_folds = []
preds_remaining_case_length_folds = []
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

    # predict case outcome
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
    
    # get predictions for each dataset
    X_train = pd.DataFrame(X_train, index=dt_train_prefixes.groupby(dataset_manager.case_id_col).first().reset_index()[dataset_manager.case_id_col])
    X_val = pd.DataFrame(X_val, index=dt_val_prefixes.groupby(dataset_manager.case_id_col).first().reset_index()[dataset_manager.case_id_col])
    X_test = pd.DataFrame(X_test, index=dt_test_prefixes.groupby(dataset_manager.case_id_col).first().reset_index()[dataset_manager.case_id_col])
    
    # predict remaining case length
    X_train["prefix_nr"] = dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"]
    X_train["case_length"] = dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["case_length"]
    X_train["orig_case_id"] = dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]
    if calibrate:
        X_val["prefix_nr"] = dt_val_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"]
        X_val["case_length"] = dt_val_prefixes.groupby(dataset_manager.case_id_col).first()["case_length"]
        X_val["orig_case_id"] = dt_val_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]
        X_train = pd.concat([X_train, X_val], axis=0)
    X_train["remaining_case_length"] = X_train["case_length"] - X_train["prefix_nr"]
    gbm_case_length = create_regression_model(X_train=X_train.drop(["remaining_case_length", "case_length", "prefix_nr", 
                                                                    "orig_case_id"], axis=1),
                                              y_train=X_train["remaining_case_length"],
                                              param=param)
    preds_case_length = gbm_case_length.predict(X_test)
    dt_preds["predicted_case_length"] = preds_case_length
    
    # predict conf delta
    X_train["predicted_proba"] = cls.predict_proba(X_train.drop(["remaining_case_length", "case_length", "prefix_nr", 
                                                                "orig_case_id"], axis=1))[:,1]

    X_train["predicted_proba_next"] = X_train.sort_values("prefix_nr", ascending=True).groupby("orig_case_id")["predicted_proba"].shift(-1)
    X_train["conf_current"] = abs(0.5 - X_train["predicted_proba"]) + 0.5
    X_train["conf_current_next"] = abs(0.5 - X_train["predicted_proba_next"]) + 0.5
    X_train["conf_diff"] = X_train["conf_current_next"] - X_train["conf_current"]
    
    X_train = X_train.dropna()
    
    gbm_conf_delta = create_regression_model(X_train=X_train.drop(["remaining_case_length", "case_length", "prefix_nr",
                                                                   "orig_case_id", "predicted_proba", "predicted_proba_next",
                                                                   "conf_current", "conf_current_next"], axis=1),
                                              y_train=X_train["conf_diff"],
                                              param=param)
    preds_conf_delta = gbm_conf_delta.predict(X_test)
    dt_preds["predicted_conf_delta"] = preds_conf_delta
    
    preds_folds.append(dt_preds.copy())
    
    
print('Optimizing parameters...')
cost_weights = []
for c01 in range(0, 100+cost_step, cost_step):
    for c10 in range(0, 100+cost_step-c01, cost_step):
        c11 = 100 - c01 - c10
        cost_weights.append((c01, c10, c11))
for c01, c10, c11 in cost_weights:
    # cost matrix
    costs = np.matrix([[lambda x: 0,
                        lambda x: c01/100.0],
                       [lambda x: c10/100.0,
                        lambda x: -c11/100.0*(x['case_length'] - x['prefix_nr'])/x['case_length']]])
    
    if alpha_variant == "parameter_free":
        space = {'coef1': hp.uniform("coef1", -1, 1),
                 'coef2': hp.uniform("coef2", -1, 1)}
        max_evals = 200
    else:
        space = {'alpha': hp.uniform("alpha", 0, 1)}
        max_evals = 50
        
    trials = Trials()
    best = fmin(evaluate_model_cost, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_params = hyperopt.space_eval(space, best)

    outfile = os.path.join(params_dir, "optimal_confs_%s_%s_%s_%s.pickle" % (dataset_name, c01, c10, c11))
    # write to file
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)
