"""
Created on Tue Oct 21 11:09:44 2025

@author: 247116 - Ming Dao Caljé 
"""

import os
import pickle
import pandas as pd

import scripts.settings as settings

def load_csv_as_dict(filename):
    """
    Load CSV file and save as pickle file in output directory.
    """
    base = os.path.splitext(filename)[0]
    out_path = os.path.join(settings.OUTPUT_DIR, f'coded_{base}.pkl')
    if os.path.exists(out_path):
        print(f'{out_path} already exists')
        return None
    db = {}

    in_path = os.path.join(settings.INPUT_DIR, filename)
    print(in_path)

    db[filename] = pd.read_csv(in_path)

    with open(out_path, 'wb') as f:
        pickle.dump(db, f)
        print('Finished saving')

    return db

def load_database(filename):
    """
    Load database from pickle file.
    """
    db = {}
    out_path = os.path.join(settings.INPUT_DIR, filename)
    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            db = pickle.load(f)
    else:
        print(f'load_database(): {out_path} does not exist')
        return None
    return db

def save_pkl(db, filename):
    """
    Save database as pickle file in output directory.
    """
    base = os.path.splitext(filename)[0]
    base = base.replace('coded_', '')
    out_path = os.path.join(settings.OUTPUT_DIR, f'decoded_{base}.pkl')
    if os.path.exists(out_path):
        print(f'{out_path} already exists, not overwriting')
    else:
        with open(out_path, 'wb') as f:
            pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'({out_path} saved)')
