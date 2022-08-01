import numpy as np
import random as random
import torch
from params import parse_args,printConfig
import os
import models

from ray import tune
from ray.tune.suggest.hebo import HEBOSearch
import ray
from ray.tune.integration.torch import DistributedTrainableCreator
from sql_writer import WriteToDatabase, get_primary_key_and_value, get_columns, merge_args_and_dict, merge_args_and_config
from statistics import mean, stdev
import socket, getpass, os
import gc
import copy
TUNE = False
def main_one(config, checkpoint_dir = None):

    ################STA|SQL|###############
    current_args = copy.deepcopy(args)
    current_args = merge_args_and_config(current_args, config)
    host_name = socket.gethostname()
    db_input_dir = {"name": ["text", host_name + os.path.split(__file__)[-1][:-3]],
                    "epoch": ["integer", None],
                    "stop_epoch": ["integer", None],
                    "seed": ["integer", None],}  #Larry: set key
    PRIMARY_KEY, PRIMARY_VALUE = get_primary_key_and_value(merge_args_and_dict(copy.deepcopy(db_input_dir), vars(current_args)))
    REFRESH = False
    OVERWRITE = True

    test_metrics = {
        "acc": None,
        "time": None,
    }
    train_metrics = {
        "acc": None,
        "time": None,
    }
    val_metrics = {
        "acc": None,
        "time": None,
    }

    TABLE_NAME = 'main_RLGL_' + current_args.task + '_' + current_args.method + '_0'
    try:
        writer = WriteToDatabase({'host': "postgres.kongfei.life", "port": "",
                                  "database": "pengliang", "user": "pengliang", "password": ""},
                                 TABLE_NAME,
                                 PRIMARY_KEY,
                                 get_columns(train_metrics, val_metrics, test_metrics),
                                 PRIMARY_VALUE,
                                 PRIMARY_VALUE,
                                 REFRESH,
                                 OVERWRITE)
        writer.init()
    except:
        print("Keys not matched in current table, pls check KEY, or network error")
        print("Change TABLE_NAME to create a new table")
    ################END|SQL|###############

    current_args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ACC_seed = []
    Time_seed = []
    for seed in range(2020,2024):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        method_fun = models.getmodel(current_args.method)
        embedder = method_fun(copy.deepcopy(current_args))

        test_acc, training_time, stop_epoch = embedder.training()

        ################STA|write one|###############
        writer_matric_seed = {'epoch': -1, "seed": seed,"test_time": training_time,"stop_epoch": stop_epoch,}
        writer.write(writer_matric_seed,
                     {
                         "test_acc": test_acc,
                     }
                     )
        ################END|write one|###############
        ACC_seed.append(test_acc)
        # St_seed.append(np.mean(test_st))
        Time_seed.append(training_time)
        torch.cuda.empty_cache()
        gc.collect()

    ################STA|write seed|###############
    writer_matric_seed = {'epoch': -2, "seed": -2,"test_time": mean(Time_seed), "stop_epoch": -2
                          }
    writer.write(writer_matric_seed,
                 {
                     "test_acc": mean(ACC_seed),
                 }
                 )
    ################END|write seed|###############
    if TUNE:
        tune.report(test_sum = mean(ACC_seed))

def main(args):
    # param set
    ################STA|set tune param|###############
    if TUNE:
        os.environ['CUDA_VISIBLE_DEVICES'] =  "1,2,4,6,7"
        ray.init(num_gpus=5)
        config = {
        'nb_epochs':tune.choice([200, 400, 800, 1000]),
        'lr':tune.choice([0.01, 0.001, 0.0005, 0.0001]),
        'wd': tune.choice([0.0001, 0.00001, 0]),
        'test_epo':tune.choice([50, 100, 200]),
        'test_lr':tune.choice([0.01, 0.001]),
        'cfg':  tune.choice([[512,256], [256,128], [128,64]]),
        'random_aug_feature': tune.choice([0.0, 0.1, 0.2, 0.5]),
        'random_aug_edge': tune.choice([0.0, 0.1, 0.2, 0.5]),
        'alpha': tune.choice([5, 1, 0.5, 0.2, 0.1, 0.02]),
        'beta': tune.choice([1, 0.5, 0.2, 0.1, 0.05, 0.01]),
        'gnn': tune.choice(["GCN","GAT"]),
        }
        # search_alg = HEBOSearch(metric='test_sum', mode='max')
        distributed_ray_run = DistributedTrainableCreator(
            main_one,
            backend='nccl',
            num_gpus_per_worker=0.5,
            num_workers=1,
        )
        tune.run(distributed_ray_run, config=config, num_samples=1000 )
        # search_alg.save('checkpoint_alg')
    else:
        config = {
            'nb_epochs': 1000,
            'lr': 0.01,
            'wd': 0.0005,
            'test_epo': 50,
            'test_lr': 0.01,
            'cfg': [16],
            'random_aug_feature': 0.1,
            'random_aug_edge': 0.0,
            'alpha': 1,
            'beta': 0.1,
            'gnn': "GCN",
        }
        main_one(config)
    ################END|set tune param|###############

if __name__ == '__main__':
    task = 'Semi'    # choice:Semi Unsup Sup Rein Noise
    method = 'Gcn'  # choice: Gcn
    dataset = 'Cora' # choice:Cora CiteSeer PubMed
    args = parse_args(task, method, dataset)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)
