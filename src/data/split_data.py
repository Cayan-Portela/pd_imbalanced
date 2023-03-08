import pprint
import pandas as pd
from load_data import read_params, read_data, clean_data_to_csv

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    RandomizedSearchCV,
    GridSearchCV,
    KFold,
    StratifiedKFold
)

# GroupTimeSeriesSplit
from mlxtend.evaluate.time_series import (
    GroupTimeSeriesSplit,
    plot_splits,
    print_cv_info,
    print_split_info
)

def mask_safra(data, safra_col, train_date, test_date):
    
    if data[safra_col].is_monotonic:
        
        mask_treino = (data[safra_col] <= pd.to_datetime(train_date))
        mask_teste  = (data[safra_col] == pd.to_datetime(test_date))

        df_mask_treino = data[mask_treino]
        df_mask_teste = data[mask_teste]

    else:
        print(f"Dados não ordenados por {safra_col}")

    return df_mask_treino, df_mask_teste

def safra_x_y(data, features, target):

    X_train = data[features]
    y_train = data[target]

    return X_train, y_train

def shuffle_x_y(x_data, y_data, test_size, seed):
   #X_shuffle_train, X_shuffle_test, y_shuffle_train, y_shuffle_test = train_test_split(X_ate_set, y_ate_set, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    
    config = read_params(config_path="params.yaml")
    
    proc_data_path = config["raw_data_config"]["processed_data_csv"]
    proc_data_config = config["proc_data_config"]
    
    safra_config = proc_data_config["safra_config"]
    shuffle_config =  proc_data_config["shuffle_config"]

    data_safra_train = safra_config["train_date"]
    data_safra_teste = safra_config["test_date"]

    safra_col = safra_config["col"]

    features = proc_data_config["model_features"]
    target = proc_data_config["target"]

    dados = read_data(proc_data_path)
    dados[safra_col] = pd.to_datetime(dados[safra_col])

    # separa safra em treino teste
    treino_safra, teste_safra = mask_safra(
        data=dados,
        safra_col=safra_col,
        train_date=data_safra_train,
        test_date=data_safra_teste
        )

    # separa safra em x e y
    safra_features = [safra_col] + features
    X_train_safra, y_train_safra = safra_x_y(data=treino_safra, features=safra_features, target=target)
    X_teste_safra, y_teste_safra = safra_x_y(data=teste_safra, features=safra_features, target=target)

    # escreve x_train e y_train
    clean_data_to_csv(data=X_train_safra, data_path=safra_config["x_train_csv"])
    clean_data_to_csv(data=y_train_safra, data_path=safra_config["y_train_csv"])
    
    # escreve x_test e y_test
    clean_data_to_csv(data=X_teste_safra, data_path=safra_config["x_test_csv"])
    clean_data_to_csv(data=y_teste_safra, data_path=safra_config["y_test_csv"])

    # separa shuffle em x e y
    X_shuffle_train, X_shuffle_test, y_shuffle_train, y_shuffle_test = shuffle_x_y(
        x_data=dados[features],
        y_data=dados[target],
        test_size=shuffle_config["test_size"],
        seed=shuffle_config["random_state"]
        )
    
    # escreve x_train e y_train
    clean_data_to_csv(data=X_shuffle_train, data_path=shuffle_config["x_train_csv"])
    clean_data_to_csv(data=y_shuffle_train, data_path=shuffle_config["y_train_csv"])
    
    # escreve x_test e y_test
    clean_data_to_csv(data=X_shuffle_test, data_path=shuffle_config["x_test_csv"])
    clean_data_to_csv(data=y_shuffle_test, data_path=shuffle_config["y_test_csv"])

    pprint.pprint(config)
    print("\n")
    print(f"X_train_safra\n{X_train_safra.head()}\n")
    print(f"X_train_safra\n{X_train_safra.tail()}\n")
    print("\n")
    print(f"X_test_safra\n{X_teste_safra.head()}")
    print("\n")
    print(f"X_train_shuffle shape: {X_shuffle_train.shape}\n{X_shuffle_train.head()}")
    print("\n")
    print(f"X_test_shuffle shape: {X_shuffle_test.shape}\n{X_shuffle_test.head()}")
    