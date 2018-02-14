import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion

import time
import os
import sys
import csv
from sys import argv
import pickle


def calculate_cost(x, costs):
    return costs[int(x['prediction']), int(x['actual'])](x)

dataset_name = argv[1]
predictions_dir = argv[2]
predictions_dir_regression = argv[3]
conf_threshold_dir = argv[4]
results_dir = argv[5]
alpha_variant = argv[6]
cost_step = int(argv[7])


# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))
    
# load predictions    
dt_preds = pd.read_csv(os.path.join(predictions_dir, "preds_%s.csv" % dataset_name), sep=";")
dt_preds2 = pd.read_csv(os.path.join(predictions_dir_regression, "preds_%s.csv" % dataset_name), sep=";")
dt_preds = dt_preds.merge(dt_preds2, on=["prefix_nr", "case_id"])
del dt_preds2


# write results to file
out_filename = os.path.join(results_dir, "results_%s_%s.csv" % (dataset_name, alpha_variant))
with open(out_filename, 'w') as fout:
    writer = csv.writer(fout, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)
    writer.writerow(["dataset", "method", "metric", "value", "c01", "c10", "c11"])

    cost_weights = []
    for c01 in range(0, 100+cost_step, cost_step):
        for c10 in range(0, 100+cost_step-c01, cost_step):
            c11 = 100 - c01 - c10
            cost_weights.append((c01, c10, c11))
    for c01, c10, c11 in cost_weights:
        # load the optimal params
        args_file = os.path.join(conf_threshold_dir, "optimal_confs_%s_%s_%s_%s.pickle" % (dataset_name, c01, c10, c11))
        try:
            with open(args_file, "rb") as fin:
                args = pickle.load(fin)
        except:
            continue

        costs = np.matrix([[lambda x: 0,
                            lambda x: c01/100.0],
                           [lambda x: c10/100.0,
                            lambda x: -c11/100.0*(x['case_length'] - x['prefix_nr'])/x['case_length']]])
    
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

        # calculate precision, recall etc.
        prec, rec, fscore, _ = precision_recall_fscore_support(dt_final.actual, dt_final.prediction, pos_label=1, average="binary")

        # calculate earliness based on the "true alarms" only
        tmp = dt_final[(dt_final.prediction == 1) & (dt_final.actual == 1)]
        earliness = (1 - (tmp.prefix_nr / tmp.case_length))

        writer.writerow([dataset_name, alpha_variant, "prec", prec, c01, c10, c11])
        writer.writerow([dataset_name, alpha_variant, "rec", rec, c01, c10, c11])
        writer.writerow([dataset_name, alpha_variant, "fscore", fscore, c01, c10, c11])
        writer.writerow([dataset_name, alpha_variant, "earliness_mean", earliness.mean(), c01, c10, c11])
        writer.writerow([dataset_name, alpha_variant, "earliness_std", earliness.std(), c01, c10, c11])

        cost = dt_final.apply(calculate_cost, costs=costs, axis=1).sum()
        writer.writerow([dataset_name, alpha_variant, "cost", cost, c01, c10, c11])
