import pandas as pd
import numpy as np
import pickle
import sys, os
import pprint

from src.data.load_data import read_params, read_data



from clf_class import ClfSwitcher

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.under_sampling import TomekLinks

# parameters of best models
from parameters import (
    # safra
    safra_best_parameters_bbagg,
    safra_best_parameters_smote,
    safra_best_parameters_tomek,
    # shuffle
    shuffle_best_parameters_bbagg,
    shuffle_best_parameters_smote,
    shuffle_best_parameters_tomek
)

# criar lista de tuples para usar no transformers
def preproc_cenario(cenario=None):
    
    # ordem colunas:
    # sexo, instrucao, idade, log_saldo_poup_avg, log_tempo_renda, log_renda_liquida, log_tempo_cp
    step_cen = {
        "pass": ("pass", "passthrough", [0]),
        "onehot_instrucao": ("cat", OneHotEncoder(), [1]),
        "std_all": ("num", StandardScaler(), [2, 3, 4, 5, 6]),
        "std_not_idade": ("num", StandardScaler(), [3, 4, 5, 6])
    }

    if cenario == 0:
        preproc_i = ColumnTransformer(
            transformers=[step_cen["pass"], step_cen["onehot_instrucao"], step_cen["std_all"]],
            remainder="drop"
            )
    elif cenario == 1:
        preproc_i = ColumnTransformer(
            transformers=[step_cen["onehot_instrucao"], step_cen["std_all"]],
            remainder="drop"
            )
    elif cenario == 2:
        preproc_i = ColumnTransformer(
            transformers=[step_cen["std_all"]],
            remainder="drop"
            )
    elif cenario == 3:
        preproc_i = ColumnTransformer(
            transformers=[step_cen["std_not_idade"]],
            remainder="drop"
            )
    else:
        print(f"Scenario not available")

    return preproc_i


def balancing_pipe(balanceamento="", preproc_i=None, best_estimator=None):
    """
    Criando pipeline
    TODO: Explicar certinho a função
    """

    if balanceamento == "bbagg":

        pipe_clf = imb_pipeline([
            ('pre_processor', preproc_i),
            ('clf', best_estimator)
        ])

        pipe_final = BalancedBaggingClassifier(
            base_estimator=pipe_clf,
            sampling_strategy='not minority',
            n_estimators=20, n_jobs=-1
        )

    elif balanceamento == "smote":
        
        pipe_final = imb_pipeline([
            ('pre_processor', preproc_i),
            ('smote', SMOTE()),
            ('clf', best_estimator)
        ])
    
    elif balanceamento == "tomek":

        tomek_link = TomekLinks(sampling_strategy="all", n_jobs=-1)

        pipe_final = imb_pipeline([
            ('pre_processor', preproc_i),
            ('smote', SMOTETomek(tomek=tomek_link)),
            ('clf', best_estimator)
        ])

    else:
        return "balanceamento (smote) inválido"
    
    return pipe_final


def run_model(final_pipe, separation, X, y):
    """
    TODO: explicar função
    """

    if separation == "safra":
        print("fitando safra")
        final_pipe.fit(X, y)

    elif separation == "shuffle":
        print("fitando shuffle")
        final_pipe.fit(X, y)

    return final_pipe

def save_pickle(fitted_pipe, separation="", pkl_file_name=""):
    """
    
    """
    pkl_file = separation +  pkl_file_name + ".pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(fitted_pipe, f)


if __name__ == '__main__':
    
    config = read_params(config_path="params.yaml")

    # _config
    safra_config = config["proc_data_config"]["safra_config"]
    shuffle_config = config["proc_data_config"]["shuffle_config"]

    # read data
    x_safra_train = read_data(safra_config["x_train_csv"])
    y_safra_train = read_data(safra_config["y_train_csv"])

    x_shuffle_train = read_data(shuffle_config["x_train_csv"])
    y_shuffle_train = read_data(shuffle_config["y_train_csv"])

    # best parameters to dict
    safra_split = {
        'safra_bbagg': safra_best_parameters_bbagg,
        'safra_smote': safra_best_parameters_smote,
        'safra_tomek': safra_best_parameters_tomek
    }

    shuffle_split = {
        'shuffle_bbagg': shuffle_best_parameters_bbagg,
        'shuffle_smote': shuffle_best_parameters_smote,
        'shuffle_tomek': shuffle_best_parameters_tomek
    }

    # combined list
    splits_agg = [safra_split, shuffle_split]
    
    # each dict
    for i_0, dict_i in enumerate(splits_agg):
        loop_split = dict_i

        # separation_balanceamento. Ex: shuffle_smote
        for i, key_i in enumerate(loop_split):
            print(f"\ni: {i}, key_i: {key_i}")
            model_configuration = loop_split[key_i]
        
            separation_i = key_i[:-6] # safra or shuffle
            balanceamento_i = key_i[-5:] #bbagg, smote or tomek

            print(f"separation_i: {separation_i}")
            print(f"balanecamento_i: {balanceamento_i}")
            
            # fixando pre_process para cada cenario (0, 1, 2, 3)
            for j, key_j in enumerate(model_configuration):
                print(f"\n\tj: {j}, key_j: {key_j}")

                pre_proc_j = preproc_cenario(cenario = j)
                
                #modelo por cenario, cenario fixo. Ex: cenario_1_modelo_0, cenario_1_modelo_1, ..., cenario1_modelo_5
                for k, key_k in enumerate(model_configuration[key_j]):
                    print(f"\t\tk: {k}, key_k: {key_k}")
                    
                    final_pipe = balancing_pipe(
                        balanceamento = balanceamento_i,
                        preproc_i = pre_proc_j,
                        best_estimator = list(key_k.values())[0] #ex: model_configuration['cenario_1'][0]['base_estimator__clf__estimator']
                    )
                    
                    fitted_pipe = run_model(
                     final_pipe=final_pipe,
                     separation=separation_i,
                     X=x_shuffle_train,
                     y=y_shuffle_train.values.ravel()
                    )
                    
                    pkl_name = key_i + "_" + key_j + "_modelo_" + str(k)
                    print(f"\t\tsalvando: {pkl_name}.pkl\n")
                    save_pickle(fitted_pipe, separation="", pkl_file_name=pkl_name)