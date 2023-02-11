import pandas as pd
import numpy as np

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
    safra_best_parameters_bbagg_cenario1,
    safra_best_parameters_bbagg_cenario2,
    safra_best_parameters_bbagg_cenario3,
    safra_best_parameters_bbagg_cenario4,

    safra_best_parameters_smote_cenario1,
    safra_best_parameters_smote_cenario2,
    safra_best_parameters_smote_cenario3,
    safra_best_parameters_smote_cenario4,

    safra_best_parameters_tomek_cenario1,
    safra_best_parameters_tomek_cenario2,
    safra_best_parameters_tomek_cenario3,
    safra_best_parameters_tomek_cenario4,

    # shuffle
    shuffle_best_parameters_bbagg_cenario1,
    shuffle_best_parameters_bbagg_cenario2,
    shuffle_best_parameters_bbagg_cenario3,
    shuffle_best_parameters_bbagg_cenario4,

    shuffle_best_parameters_smote_cenario1,
    shuffle_best_parameters_smote_cenario2,
    shuffle_best_parameters_smote_cenario3,
    shuffle_best_parameters_smote_cenario4,

    shuffle_best_parameters_tomek_cenario1,
    shuffle_best_parameters_tomek_cenario2,
    shuffle_best_parameters_tomek_cenario3,
    shuffle_best_parameters_tomek_cenario4
)

def smote_pipe(balanceamento="", cenario=None, best_estimator=None):
    """
    Criando pipeline
    TODO: Explicar certinho a função
    """
    # ordem colunas:
    # sexo, instrucao, idade, log_saldo_poup_avg, log_tempo_renda, log_renda_liquida, log_tempo_cp

    if cenario == 1:
        preproc_i = ColumnTransformer(
                            transformers=[
                                ("pass", "passthrough", 0),
                                ("cat", OneHotEncoder(), 1),
                                ("num", StandardScaler(), [2, 3, 4, 5, 6])
                            ],
                            remainder="drop"
                        )
        
    elif cenario == 2:
        # remove sexo
        preproc_i = ColumnTransformer(
                            transformers=[
                                #("pass", "passthrough", 0),
                                ("cat", OneHotEncoder(), 1),
                                ("num", StandardScaler(), [2, 3, 4, 5, 6])
                            ],
                            remainder="drop"
                    )

    elif cenario == 3:
        # remove sexo e instrucao
        preproc_i = ColumnTransformer(
                            transformers=[
                                #("pass", "passthrough", 0),
                                #("cat", OneHotEncoder(), 1),
                                ("num", StandardScaler(), [2, 3, 4, 5, 6])
                            ],
                            remainder="drop"
                    )
        
    elif cenario == 4:
        # remove sexo, instrucao e idade
        preproc_i = ColumnTransformer(
                            transformers=[
                                #("pass", "passthrough", 0),
                                #("cat", OneHotEncoder(), 1),
                                #("num", StandardScaler(), [2, 3, 4, 5, 6])
                                ("num", StandardScaler(), [3, 4, 5, 6])
                            ],
                            remainder="drop"
                    )
        
    else:
        print("nenhum celario escolhido.")


    if balanceamento == "none":

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
            ('smote', SMOTE(n_jobs=-1)),
            ('clf', best_estimator)
        ])
    
    elif balanceamento == "smote_tomek":

        tomek_link = TomekLinks(sampling_strategy="all", n_jobs=-1)

        pipe_final = imb_pipeline([
            ('pre_processor', preproc_i),
            ('smote', SMOTETomek(tomek=tomek_link, n_jobs=-1)),
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

    pkl_file = separation + "model_.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(final_pipe, f)
