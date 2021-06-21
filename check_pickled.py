import pickle
import os
from pathlib import Path
from pandas import DataFrame 

def compare_pickle_files(path1, path2):

    batch_obj = pickle.load(open(path1, 'rb'))
    pipeline_obj = pickle.load(open(path2, 'rb'))
    fbase = os.path.basename(path1) 

    if isinstance(batch_obj, DataFrame):

        if (fbase == 'df_ia_agg_scored.pkl' or fbase == 'df_ia_agg_scored.2.pkl' 
            or fbase == 'df_ia_agg_scored_explainer.pkl'):
            batch_obj = clean_df_ia_agg_scored(batch_obj)
            pipeline_obj = clean_df_ia_agg_scored(pipeline_obj)

        if not batch_obj.equals(pipeline_obj):
            print('ERROR: DataFrames do not match.')

    else:
        print('WARNING: this object is not a DataFrame. Object type is', type(batch_obj))

def clean_df_ia_agg_scored(df):
    if 'explainer_results' in df.columns:
        df = df.drop(columns=['explainer_results'])
    """
    if 'exp_list' in df.columns:
        df = df.drop(columns=['exp_list'])
    if 'S/M/L term' in df.columns:
        df = df.drop(columns=['S/M/L term'])
        """
    return df

PICKLED_BATCH = 'C:/Users/guzik/Desktop/Anton/pickled/batch'
PICKLED_PIPELINE = 'C:/Users/guzik/Desktop/Anton/pickled/pipeline'

for f in os.listdir(PICKLED_BATCH):
    
    if not f.endswith('.pkl'):
        continue

    print('Checking pickled object', f)
    p = Path(PICKLED_PIPELINE + '/' + f)
    
    if not p.is_file():    
        print('ERROR: File', PICKLED_PIPELINE + '/' + f, 'not found.')
    else:
        compare_pickle_files(PICKLED_BATCH + '/' + f, PICKLED_PIPELINE + '/' + f)
