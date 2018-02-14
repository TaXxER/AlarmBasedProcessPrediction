import EncoderFactory
from DatasetManager import DatasetManager
from calibration_wrappers import LGBMCalibrationWrapper

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
import time
import os
import sys
from sys import argv
import pickle

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt


def calculate_cost(x, costs):
    return costs[int(x['prediction']), int(x['actual'])](x)

def evaluate_model_cost(args):
    total_cost = 0
    # trigger alarms according to conf_threshold
    dt_final = pd.DataFrame()
    unprocessed_case_ids = set(dt_preds.case_id.unique())
    for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
        tmp["predicted_proba_next"] = tmp.predicted_proba + abs(tmp.predicted_conf_delta)
            
        expected_cost_now_alert = (tmp.predicted_proba * costs[1,1]({"prefix_nr": nr_events,
                                                                     "case_length": nr_events+tmp.case_length}) +
                                   (1-tmp.predicted_proba) * costs[1,0]({}))
        expected_cost_now_no_alert = (tmp.predicted_proba * costs[0,1]({}) +
                                      (1-tmp.predicted_proba) * costs[0,0]({}))
        expected_cost_next_alert = (tmp.predicted_proba_next * costs[1,1]({"prefix_nr": nr_events+1,
                                                                           "case_length": nr_events+tmp.case_length}) +
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
preds_dir = argv[7]

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
# read the data
dataset_manager = DatasetManager(dataset_name)

# load pre-tuned lgbm params
with open(optimal_params_filename, "rb") as fin:
    param = pickle.load(fin)
param['metric'] = ['auc', 'binary_logloss']
param['objective'] = 'binary'
param['verbosity'] = -1
    

dt_preds = pd.read_csv(os.path.join(preds_dir, "preds_train_%s.csv" % dataset_name), sep=";")
case_lengths = dt_preds.groupby("case_id").prefix_nr.max().reset_index()
case_lengths.columns = ["case_id", "case_length"]
dt_preds = dt_preds.merge(case_lengths)

dt_preds["predicted_proba_next"] = dt_preds.sort_values("prefix_nr", ascending=True).groupby("case_id")["predicted_proba"].shift(-1)
dt_preds["predicted_conf_delta"] = abs(dt_preds["predicted_proba"] - dt_preds["predicted_proba_next"])
    
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
