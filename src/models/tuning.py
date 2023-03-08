import pandas as pd
import numpy as np
import pickle
import sys, os
import pprint

from src.data.load_data import read_params, read_data
from src.models.smote import preproc_cenario, balancing_pipe
from src.models.parameters import (
    parameters_bagg,
    parameters_smote
)

from src.models.clf_class import ClfSwitcher

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, RandomizedSearchCV, GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from mlxtend.evaluate.time_series import (
    GroupTimeSeriesSplit,
    plot_splits,
    print_cv_info,
    print_split_info
)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def mask_safra(x):
    if x == pd.to_datetime("2020-05-01"):
        return 0
    elif x == pd.to_datetime("2020-06-01"):
        return 1
    elif x == pd.to_datetime("2020-07-01"):
        return 2
    elif x == pd.to_datetime("2020-08-01"):
        return 3
    elif x == pd.to_datetime("2020-09-01"):
        return 4
    else:
        return 9


if __name__ == '__main__':

    config = read_params(config_path="params.yaml")
    
    safra_config = config["proc_data_config"]["safra_config"]
    shuffle_config = config["proc_data_config"]["shuffle_config"]

    #
    shuffle_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)

    hyper_par = {
        'bbagg': parameters_bagg,
        #'smote': parameters_smote,
        #'tomek': parameters_smote
    }
    
    x_shuffle_train = read_data(shuffle_config["x_train_csv"])
    y_shuffle_train = read_data(shuffle_config["y_train_csv"])

    x_safra_train = read_data(safra_config["x_train_csv"])
    y_safra_train = read_data(safra_config["y_train_csv"])

    grid_bbagg = []
    grid_smote = []

###############

    # shuffle
    if False:
        validacao = 'shuffle'
        x_grid = x_shuffle_train
        y_grid = y_shuffle_train.default_ate_12m

        # base_estimator__clf ou clf__est imator
        for k in hyper_par:
            # logistica, knn, decision_tree, rf ou boosting
            for i, values in enumerate(hyper_par[k]):
                # todas var; sem sexo; sem idade; ...
                for cenario_j in [0, 1, 2, 3]:
                    print(f"i: {i}, k: {k}, cenario: {cenario_j} \n\t {values}\n") 

                    j_pre_proc = preproc_cenario(cenario = cenario_j)
                    j_pipe = balancing_pipe(
                        balanceamento = k,
                        preproc_i = j_pre_proc,
                        best_estimator = ClfSwitcher()
                        )
                    
                    grid_ith = GridSearchCV(
                        estimator=j_pipe,
                        param_grid=values,
                        scoring='recall',
                        cv=shuffle_cv,
                        verbose=3,
                        n_jobs=-1
                        ).fit(x_grid, y_grid)
                    
                    pkl_file_name =  + 'validacao_' + k + '_' 'mod' + str(i) + '_' + 'cenario' + str(cenario_j) + ".pkl"

                    with open(pkl_file_name, 'wb') as f:
                        pickle.dump(grid_ith, f)

    # safra
    if True:

        validacao = 'safra'
        x_safra_train.safra_contrato = x_safra_train.safra_contrato.apply(pd.to_datetime)
        
        x_grid = x_safra_train.drop(columns='safra_contrato')
        y_grid = y_safra_train.default_ate_12m

        groups_cv = x_safra_train.safra_contrato.apply(mask_safra)
        groups_cv = groups_cv[ groups_cv != 9]

        cv_args = {"test_size": 1, "n_splits": 4, "window_type": "expanding"}
        custom_cv = GroupTimeSeriesSplit(**cv_args)

        # base_estimator__clf ou clf__est imator
        for k in hyper_par:
            # logistica, knn, decision_tree, rf ou boosting
            for i, values in enumerate(hyper_par[k]):

                    # todas var; sem sexo; sem idade; ...
                    for cenario_j in [0, 1, 2, 3]:
                        print(f"i: {i}, k: {k}, cenario: {cenario_j} \n\t {values}\n") 

                        j_pre_proc = preproc_cenario(cenario = cenario_j)
                        j_pipe = balancing_pipe(
                            balanceamento = k,
                            preproc_i = j_pre_proc,
                            best_estimator = ClfSwitcher()
                            )
                        
                        grid_ith = GridSearchCV(
                            estimator=j_pipe,
                            param_grid=values,
                            scoring='recall',
                            cv=custom_cv,
                            verbose=3,
                            n_jobs=-1
                            ).fit(x_grid, y_grid,  groups=groups_cv)
                        
                        pkl_file_name = validacao + '_' + k + '_' 'mod' + str(i) + '_' + 'cenario' + str(cenario_j) + ".pkl"
                        
                        print(f"Salvando {pkl_file_name}")
                        with open(pkl_file_name, 'wb') as f:
                                pickle.dump(grid_ith, f)

