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
        
    total_cost = dt_final.apply(calculate_cost, costs=costs, axis=1).sum()

    return {'loss': total_cost, 'status': STATUS_OK, 'model': dt_final}


print('Preparing data...')
start = time.time()

dataset_name = argv[1]
preds_dir = argv[2]
params_dir = argv[3]
calibrate = bool(argv[4])
cost_step = int(argv[5])

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
# read the data
dataset_manager = DatasetManager(dataset_name)
    
dt_preds = pd.read_csv(os.path.join(preds_dir, "preds_train_%s.csv" % dataset_name), sep=";")

print('Optimizing parameters...')
#cost_weights = [(1,1,-5), (5,1,-1), (1,5,-1), (5,1,-5), (1,5,-5), (5,5,-1), (1,1,-1)]
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
    
    space = {'conf_threshold': hp.uniform("conf_threshold", 0, 1)}
    trials = Trials()
    best = fmin(evaluate_model_cost, space, algo=tpe.suggest, max_evals=50, trials=trials)

    best_params = hyperopt.space_eval(space, best)

    outfile = os.path.join(params_dir, "optimal_confs_%s_%s_%s_%s.pickle" % (dataset_name, c01, c10, c11))
    # write to file
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)
