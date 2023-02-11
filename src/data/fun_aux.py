
import pandas as pd

def junta_superior(x):
    if x in ['superior', 'pos_grad']:
        return 'superior_pos'
    else:
        return x

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