import pprint
import yaml
import argparse
import numpy as pd
import pandas as pd

import sys,os
#sys.path.append(os.getcwd())

from src.data.fun_aux import junta_superior, mask_safra

def read_params(config_path):
    """
    Read parameters from the params.yaml file
    :param config_path: params.yaml location
    :return: paramteres as dictionary
    """
    with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
    return config

def read_data(data_path):
      df = pd.read_csv(data_path)
      return df

def drop_na(data):
      data = data.dropna().reset_index(drop=True)
      return data

def drop_col(data, col):
      data = data.drop(columns=col)
      return data

def sort_safra(data, col):
      data = data.sort_values(by=col, ignore_index=True)
      return data

def mutate_col(data, col, fun):
      data[col] = data[col].apply(fun)
      return data

def map_sexo(data, col):
      map_sexo = {"fem": 0, "masc": 1}
      data[col] = data[col].map(map_sexo)
      return data

def clean_data(data, safra_col):
      """
      Limpeza geral e retorna dados prontos para o split
      """
      config = read_params(config_path="params.yaml")
      
      features_num = config["raw_data_config"]["model_features"]["numeric"]
      features_bin = config["raw_data_config"]["model_features"]["binary"]
      features_cat = config["raw_data_config"]["model_features"]["categorical"]
      
      target_ = config["raw_data_config"]["target"]
      target = [target_]

      data = drop_na(data)
      data = drop_col(data, col="Unnamed: 0")
      data = sort_safra(data, col="safra_contrato")
      #print(data.head())
      data = mutate_col(data, "instrucao", junta_superior)

      data = map_sexo(data,col="sexo")

      for col in features_cat:
            data[col] = data[col].astype("category")

      # Adicionando safra como primeira coluna
      #safra_col_ = proc_data_config["safra_config"]["col"]
      safra_col_lista = [safra_col]

      ordem_colunas = features_bin + features_cat + features_num + target
      data = data[safra_col_lista + ordem_colunas]

      return data

def clean_data_to_csv(data, data_path):
      data.to_csv(data_path, index=False)

if __name__ == "__main__":
      #
      config = read_params(config_path="params.yaml")
      data_path = config["raw_data_config"]["raw_data_csv"]
      proc_data_path = config["raw_data_config"]["processed_data_csv"]

      proc_data_config = config["proc_data_config"]
      safra_col_ = proc_data_config["safra_config"]["col"]

      dados = read_data(data_path)
      print(f"dados raw:\n")
      print(dados.head())

      
      dados_clean = clean_data(dados, safra_col=safra_col_)

      clean_data_to_csv(data=dados_clean, data_path=proc_data_path)
      print(os.getcwd())
      pprint.pprint(config)
      print("\n")
      print(f"head()\n")
      #print(dados_clean.head())
      print(f"\n")
      print(f"tail()\n")
      print(sys.path)
      #print(dados_clean.tail())
