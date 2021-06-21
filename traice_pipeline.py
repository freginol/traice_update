import numpy as np
import pandas as pd
import glob
import pickle
import os
from math import ceil

# TEST
PICKLED_DIR = 'C:/Users/guzik/Desktop/Anton/pickled/pipeline'

################################### LOAD ###################################

## Feature creation

# Reading Tables
os.chdir('C:/Users/guzik/Desktop/Anton/traice_inputs')
table_paths = glob.glob('LEX*csv')
this_run_id = '02'

# Read all tables from directory
tables = []
for tbf in table_paths:
    tables.append(pd.read_csv(tbf, encoding='latin1'))

## Defining table names and bridging account and trades data

# get names of tables for easy indexing
table_names = list(map(lambda x: x.split('\\')[-1][:-4],table_paths))

# all trades
trades = tables[table_names.index('LEX_TRADE')]

# table bridging accounts and trades tables
acct_trade_bridge = tables[table_names.index('LEX_ACCT_TRADE_BRIDGE')]

# accounts
acct = tables[table_names.index('LEX_ACCT')]

# trade indicators
trd_inds_all = tables[table_names.index('LEX_TRD_IND')]

# unique trades within trade indicators
trd_inds = tables[table_names.index('LEX_TRD_IND')].drop_duplicates('TRD_TRADE_ID')

# complaints
complaints = tables[table_names.index('LEX_COMPLAINTS')]

# gross revenues
gross_revs = tables[table_names.index('LEX_GROSS_IA_INCOME')]

# TEST
print('Pickling after load')
pickle.dump(trades, open(PICKLED_DIR + '/trades.pkl', 'wb'))
pickle.dump(acct_trade_bridge, open(PICKLED_DIR + '/acct_trade_bridge.pkl', 'wb'))
pickle.dump(acct, open(PICKLED_DIR + '/acct.pkl', 'wb'))
pickle.dump(trd_inds_all, open(PICKLED_DIR + '/trd_ind.pkl', 'wb'))
pickle.dump(complaints, open(PICKLED_DIR + '/complaints.pkl', 'wb'))
pickle.dump(gross_revs, open(PICKLED_DIR + '/gross_revs.pkl', 'wb'))

################################### MERGE ###################################

# merging the account and trades tables
acct_bridge = pd.merge(acct,acct_trade_bridge,left_on=['BIZ_DATE','ACCT_ID'],right_on=['ACCT_DATE','ACCT_ID'],how='inner')
acct_trades = pd.merge(acct_bridge,trades,on=['ACCT_ID','TRD_BIZ_DATE'],how='inner')

## Adding month column to account trades

acct_trades['TRD_MONTH'] = acct_trades['TRD_BIZ_DATE'].astype('str').apply(lambda x: x[x.index('-') + 1:])

## Converting flags into indicator columns

# reduce columns to consume less memory
keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'TRADE_INDICATOR_ID_DESC', 'WM_PHYSICAL_BRANCH_ID',
             'WM_PHY_BRANCH_REGION']
df_all_flags_merge = pd.merge(left=acct_trades, right=trd_inds_all, on='TRD_TRADE_ID', how='left')[keep_cols]

# get broken out columns for flags
dummies = pd.get_dummies(df_all_flags_merge['TRADE_INDICATOR_ID_DESC'])
dummy_cols = dummies.columns

# merge table and broken out flag columns
df_all_flags_dummies_merge = pd.merge(df_all_flags_merge, dummies, left_index=True, right_index=True, how='left')

keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION']
flagged_extended = df_all_flags_dummies_merge.groupby(keep_cols)[dummy_cols].sum().reset_index()

# read in descriptions for the various flags, store in dictionary
temp_df = pd.read_excel('trade indicator.xlsx').iloc[:-1, :][['TRADE_INDICATOR_DESC', 'Short Description']]
temp_df.index = temp_df['TRADE_INDICATOR_DESC']
del temp_df['TRADE_INDICATOR_DESC']
code_dict = temp_df.to_dict()['Short Description']

# TEST
print('Pickling after merge')
pickle.dump(acct_trades, open(PICKLED_DIR + '/acct_trades.pkl', 'wb'))
pickle.dump(flagged_extended, open(PICKLED_DIR + '/flagged_extended.pkl', 'wb'))
pickle.dump(dummy_cols, open(PICKLED_DIR + '/dummy_cols.pkl', 'wb'))
pickle.dump(code_dict, open(PICKLED_DIR + '/code_dict.pkl', 'wb'))

################################### BREAKOUT ###################################

## Pro Trades

keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'PRO_ACCOUNT', 'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION']
pro_trades = acct_trades[acct_trades['PRO_ACCOUNT'] == 'PRO'][keep_cols]

## Cancelled Trades

keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'CANCEL_INDICATOR', 'WM_PHYSICAL_BRANCH_ID',
             'WM_PHY_BRANCH_REGION']
cancelled_trades = acct_trades[acct_trades['CANCEL_INDICATOR'] == 'CXL'][keep_cols]

## Reversals

# calculate where net buy / sell quantity == 0
ia_branch_region_bridge = acct_trades[['IA_NAME', 'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION']].drop_duplicates(
    'IA_NAME')
acct_trades['Signed_Quantity'] = acct_trades['BUY_SELL_INDICATOR'].apply(lambda x: 1 if x == 'B' else -1) * acct_trades[
    'QUANTITY']
trd_grp2 = acct_trades.groupby(['TRD_BIZ_DATE', 'TRD_MONTH', 'IA_NAME', 'ACCT_ID', 'SEC_SECURITY_ID'])[
    'Signed_Quantity'].sum().reset_index()
reversals = trd_grp2[trd_grp2['Signed_Quantity'] == 0].groupby(['TRD_BIZ_DATE', 'TRD_MONTH', 'IA_NAME'])[
    'Signed_Quantity'].count().reset_index()
reversals['TRD_TRADE_ID'] = reversals['Signed_Quantity']
reversals = pd.merge(reversals, ia_branch_region_bridge, on='IA_NAME', how='inner')

## Complaints

import datetime

# convert month to same format as other month columns
complaints['TRD_MONTH'] = complaints['DATE_RECEIVED'].apply(
    lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').strftime('%b-%y').upper())
complaints = pd.merge(complaints, ia_branch_region_bridge, on='IA_NAME', how='inner')
complaints['TRD_TRADE_ID'] = complaints.index

## KYC Changes

import hashlib

# calculate number of kyc changes by taking a concatenation of all of the KYC columns and counting the number
# of distinct entries per account, then rolling up to determine the number of clients by IA with more than
# two changes
kyc_cols = [col for col in acct.columns if col[:3] == 'KYC']
kyc_only = acct_trades[kyc_cols]
acct_trades['KYC_Hash'] = kyc_only.astype(str).sum(axis=1).apply(
    lambda x: hashlib.sha256(x.encode('ascii')).hexdigest())
acct_kyc_12m = acct_trades.groupby(['ACCT_ID', 'IA_NAME', 'KYC_Hash'])['BIZ_DATE'].count().reset_index().groupby(
    ['ACCT_ID', 'IA_NAME'])['KYC_Hash'].count() - 1
acct_kyc_12m = acct_kyc_12m[acct_kyc_12m > 2].reset_index().groupby('IA_NAME')['KYC_Hash'].count().reset_index()

# excluded for now since no kyc changes within the latest month
# acct_kyc_tm = acct_trades[acct_trades['TRD_MONTH']=='JAN-17'].groupby(['ACCT_ID','IA_NAME','KYC_Hash'])['BIZ_DATE'].count().reset_index().groupby(['ACCT_ID','IA_NAME'])['KYC_Hash'].count()-1

## Traded under different IA than account IA

keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION']
traded_under_other_ia = acct_trades[acct_trades['IA_NAME'] != acct_trades['TRADE_IA_NAME']][keep_cols]

# TEST
print('Pickling after breakout')
# acct_trades was changed by get_reversals() and get_kyc_changes() and needs to be saved again
pickle.dump(acct_trades, open(PICKLED_DIR + '/acct_trades.2.pkl', 'wb'))
# complaints was changed by get_complaints() and needs to be saved again
pickle.dump(complaints, open(PICKLED_DIR + '/complaints.2.pkl', 'wb'))
pickle.dump(ia_branch_region_bridge, open(PICKLED_DIR + '/ia_branch_region_bridge.pkl', 'wb'))
pickle.dump(pro_trades, open(PICKLED_DIR + '/pro_trades.pkl', 'wb'))
pickle.dump(cancelled_trades, open(PICKLED_DIR + '/cancelled_trades.pkl', 'wb'))
pickle.dump(reversals, open(PICKLED_DIR + '/reversals.pkl', 'wb'))
pickle.dump(acct_kyc_12m, open(PICKLED_DIR + '/acct_kyc_12m.pkl', 'wb'))
pickle.dump(traded_under_other_ia, open(PICKLED_DIR + '/traded_under_other_ia.pkl', 'wb'))

################################### AGGREGATE ###################################

## Create IA Aggregations

def agg_by_ia(df):
    '''
    Aggregate the count of the TRD_TRADE_ID column for this month, and 12 month median value for each IA.
    TRD_TRADE_ID is not always the trade id in current usage
    '''
    cols_to_keep = ['IA_NAME', 'TRD_TRADE_ID']
    trd_agg = df.groupby(['TRD_MONTH', 'IA_NAME'])['TRD_TRADE_ID'].count().reset_index()

    trd_agg_tm = trd_agg[trd_agg['TRD_MONTH'] == 'JAN-17'].groupby('IA_NAME')['TRD_TRADE_ID'].sum().reset_index()[
        cols_to_keep]
    trd_agg_12m = (trd_agg.groupby('IA_NAME')['TRD_TRADE_ID'].median()).reset_index()[cols_to_keep]

    return trd_agg_tm, trd_agg_12m

def complaints_by_ia(df):
    '''
    Aggregate sum of complaints for this month and all time
    '''
    cols_to_keep = ['IA_NAME', 'TRD_TRADE_ID']
    trd_agg = df.groupby(['TRD_MONTH', 'IA_NAME'])['TRD_TRADE_ID'].count().reset_index()

    trd_agg_tm = trd_agg[trd_agg['TRD_MONTH'] == 'JAN-17'].groupby('IA_NAME')['TRD_TRADE_ID'].sum().reset_index()[
        cols_to_keep]
    trd_agg_ever = (trd_agg.groupby('IA_NAME')['TRD_TRADE_ID']).sum().reset_index()[cols_to_keep]

    return trd_agg_tm, trd_agg_ever

def agg_by_ia_sum(df, sum_col):
    '''
    Same agg agg_by_ia but using sum instead of count
    '''
    cols_to_keep = ['IA_NAME', sum_col]
    trd_agg = df.groupby(['TRD_MONTH', 'IA_NAME'])[sum_col].sum().reset_index()

    trd_agg_tm = trd_agg[trd_agg['TRD_MONTH'] == 'JAN-17'].groupby('IA_NAME')[sum_col].sum().reset_index()[cols_to_keep]
    trd_agg_12m = (trd_agg.groupby('IA_NAME')[sum_col].median()).reset_index()[cols_to_keep]

    return trd_agg_tm, trd_agg_12m

def merge_by_ia(df_left, df_right, new_col_name='AggValue', col_to_retrieve='TRD_TRADE_ID'):
    '''
    left join an IA aggregate dataframe(left) with another dataframe (right) and rename new column
    '''
    df_ia_agg = pd.merge(left=df_left, right=df_right, on='IA_NAME', how='left')
    df_ia_agg[new_col_name] = df_ia_agg[col_to_retrieve]
    df_ia_agg[new_col_name] = df_ia_agg[new_col_name].fillna(0)
    del df_ia_agg[col_to_retrieve]

    return df_ia_agg

def agg_by_other(trades, col_to_agg_by):
    '''
    Aggregate using another columns (such as region or branch)
    '''
    cols_to_keep = ['IA_NAME', 'TRD_TRADE_ID']
    trd_agg1 = trades.groupby(['TRD_MONTH', col_to_agg_by, 'IA_NAME'])['TRD_TRADE_ID'].count().reset_index()
    trd_agg = trd_agg1.groupby(['TRD_MONTH', col_to_agg_by])['TRD_TRADE_ID'].median().reset_index()

    trd_agg_tm = trd_agg[trd_agg['TRD_MONTH'] == 'JAN-17']
    trd_agg_tm = pd.merge(trd_agg_tm, ia_branch_region_bridge, on=col_to_agg_by, how='inner')[cols_to_keep]
    trd_agg_12m = trd_agg1.groupby(col_to_agg_by)['TRD_TRADE_ID'].median().reset_index()
    trd_agg_12m = pd.merge(trd_agg_12m, ia_branch_region_bridge, on=col_to_agg_by, how='inner')[cols_to_keep]
    return trd_agg_tm, trd_agg_12m

def agg_by_other_sum(trades, col_to_agg_by, sum_col):
    '''
    Same as agg_by_other but using sum instead of count
    '''
    cols_to_keep = ['IA_NAME', sum_col]
    trd_agg1 = trades.groupby(['TRD_MONTH', col_to_agg_by, 'IA_NAME'])[sum_col].sum().reset_index()
    trd_agg = trd_agg1.groupby(['TRD_MONTH', col_to_agg_by])[sum_col].median().reset_index()

    trd_agg_tm = trd_agg[trd_agg['TRD_MONTH'] == 'JAN-17']
    trd_agg_tm = pd.merge(trd_agg_tm, ia_branch_region_bridge, on=col_to_agg_by, how='inner')[cols_to_keep]
    trd_agg_12m = trd_agg1.groupby(col_to_agg_by)[sum_col].median().reset_index()
    trd_agg_12m = pd.merge(trd_agg_12m, ia_branch_region_bridge, on=col_to_agg_by, how='inner')[cols_to_keep]
    return trd_agg_tm, trd_agg_12m

def create_merged_cols(df_ia_agg, df_to_agg, new_col_name='AggValue', sum_col=None):
    '''
    Create columns for this month, twelve month median, and the same for branch and region values
    Note: branch / region currently commented out to exclude this functionality since it was adding too many features
    '''
    suffix_tm = '_TM'
    suffix_12m = '_12M'
    col_tm = new_col_name + suffix_tm
    col_12m = new_col_name + suffix_12m
    col_branch_tm = col_tm + '_BRANCH'
    col_branch_12m = col_12m + '_BRANCH'
    col_region_tm = col_tm + '_REGION'
    col_region_12m = col_12m + '_REGION'
    if 'Flagged' in new_col_name or new_col_name == 'Commission':

        agg_tm, agg_12m = agg_by_ia_sum(df_to_agg, sum_col)
        branch_agg_tm, branch_agg_12m = agg_by_other_sum(df_to_agg, col_to_agg_by='WM_PHYSICAL_BRANCH_ID',
                                                         sum_col=sum_col)
        region_agg_tm, region_agg_12m = agg_by_other_sum(df_to_agg, col_to_agg_by='WM_PHY_BRANCH_REGION',
                                                         sum_col=sum_col)
        col_to_retrieve = sum_col
    else:
        if new_col_name == 'Complaints':
            agg_tm, agg_12m = complaints_by_ia(df_to_agg)
        else:
            agg_tm, agg_12m = agg_by_ia(df_to_agg)
        branch_agg_tm, branch_agg_12m = agg_by_other(df_to_agg, col_to_agg_by='WM_PHYSICAL_BRANCH_ID')
        region_agg_tm, region_agg_12m = agg_by_other(df_to_agg, col_to_agg_by='WM_PHY_BRANCH_REGION')
        col_to_retrieve = 'TRD_TRADE_ID'
    df_agg_cur = merge_by_ia(df_ia_agg, agg_tm, col_tm, col_to_retrieve)
    df_agg_cur = merge_by_ia(df_agg_cur, agg_12m, col_12m, col_to_retrieve)

    '''
    # region and branch related - excluded for now
    df_agg_cur = merge_by_ia(df_agg_cur,branch_agg_tm,col_branch_tm,col_to_retrieve)
    df_agg_cur = merge_by_dia(df_agg_cur,branch_agg_12m,col_branch_12m,col_to_retrieve)

    df_agg_cur = merge_by_ia(df_agg_cur,region_agg_tm,col_region_tm,col_to_retrieve)
    df_agg_cur = merge_by_ia(df_agg_cur,region_agg_12m,col_region_12m,col_to_retrieve)

    if not new_col_name.lower() == 'complaints':
        df_agg_cur[new_col_name + '_TM_VS_12M'] = df_agg_cur[col_tm] - df_agg_cur[col_12m]
        df_agg_cur[new_col_name + '_TM_VS_MEDIAN'] = df_agg_cur[col_tm] - df_agg_cur[col_tm].median()
        df_agg_cur[new_col_name + '_12M_VS_MEDIAN'] = df_agg_cur[col_12m] - df_agg_cur[col_12m].median()
​
        df_agg_cur[new_col_name + '_TM_VS_BRANCH_TM'] = df_agg_cur[col_tm] - df_agg_cur[col_branch_tm]
        df_agg_cur[new_col_name + '_12M_VS_BRANCH_12M'] = df_agg_cur[col_12m] - df_agg_cur[col_branch_12m]
        df_agg_cur[new_col_name + '_TM_VS_REGION_TM'] = df_agg_cur[col_tm] - df_agg_cur[col_region_tm]
        df_agg_cur[new_col_name + '_12M_VS_REGION_12M'] = df_agg_cur[col_12m] - df_agg_cur[col_region_12m]

    del df_agg_cur[col_branch_tm]
    del df_agg_cur[col_branch_12m]
    del df_agg_cur[col_region_tm]
    del df_agg_cur[col_region_12m]
    '''

    if new_col_name.lower() == 'complaints':
        df_agg_cur.columns = [el.replace('12M', 'AllTime') for el in list(df_agg_cur.columns)]

    return df_agg_cur


## Created the aggregated dataframe by IA

# create dataframe with all unique Account level IAs to start
df_ia_agg = pd.DataFrame()
df_ia_agg['IA_NAME'] = acct_trades['IA_NAME'].unique()

# add number of trades
df_ia_agg = create_merged_cols(df_ia_agg, acct_trades, 'Trades')

# add each flag type
for col in dummy_cols:
    df_ia_agg = create_merged_cols(df_ia_agg, flagged_extended, f'Flagged_{code_dict[col]}', col)

# add total flagged trades (excluded for now since each flag type is included separately)
# df_ia_agg = create_merged_cols(df_ia_agg,flagged_trades,'Flagged_Trades')

# add number of pro trades
df_ia_agg = create_merged_cols(df_ia_agg, pro_trades, 'Pro_Trades')

# add number of cancelled trades
df_ia_agg = create_merged_cols(df_ia_agg, cancelled_trades, 'Cancelled_Trades')

# add number of complaints
df_ia_agg = create_merged_cols(df_ia_agg, complaints, 'Complaints')

# add commission amounts
df_ia_agg = create_merged_cols(df_ia_agg, acct_trades, 'Commission', 'TRD_COMMISSION')

# add trades under different IA
df_ia_agg = create_merged_cols(df_ia_agg, traded_under_other_ia, 'Trades_Under_Different_IA')

# add number of clients with more than 2 KYC changes
df_ia_agg = merge_by_ia(df_ia_agg, acct_kyc_12m, 'Clients_With_More_Than_One_KYC_Change', 'KYC_Hash')

# cleanup and standardization (convert values to mean 0, std 1)
df_ia_agg.index = df_ia_agg['IA_NAME']
del df_ia_agg['IA_NAME']
df_ia_agg_std = (df_ia_agg - df_ia_agg.mean(axis=0)) / df_ia_agg.std(axis=0)

# replace any missing values with 0
df_ia_agg_std = df_ia_agg_std.fillna(0)

# intermediate output of dataframe
df_ia_agg_std.to_excel(PICKLED_DIR + '/' + f'IA_Aggregations_{this_run_id}.xls')

# TEST
print('Pickling after aggregation')
pickle.dump(df_ia_agg, open(PICKLED_DIR + '/df_ia_agg.pkl', 'wb'))
pickle.dump(df_ia_agg_std, open(PICKLED_DIR + '/df_ia_agg_std.pkl', 'wb'))

################################### FIT ###################################

## Model and Explainer

## Remove any non - populated columns

for col in list(df_ia_agg_std.isnull().sum()[df_ia_agg_std.isnull().sum() > 1].index):
    del df_ia_agg_std[col]

def score_fn(df):
    '''
    Return a score between 0 and 1 with exponential decrease
    COEF can be tuned to result in different score distributions
    '''
    COEF = 10
    dfn = iforest.decision_function(df)

    return 1 / (1 + np.exp(COEF * dfn))

from sklearn.ensemble import IsolationForest

# produce isolation forest model and fit to dataframe
#iforest = IsolationForest(n_estimators=1000, contamination=0.05)
iforest = IsolationForest(n_estimators=1000, contamination=0.05, random_state=123)
iforest.fit(df_ia_agg_std)

# create new dataframe with output scores
df_ia_agg_scored = df_ia_agg_std.copy()

# get raw score from isolation forest model
df_ia_agg_scored['score'] = iforest.decision_function(df_ia_agg_std)

# get custom score between 0 and 1 using scoring function
df_ia_agg_scored['risk_score'] = score_fn(df_ia_agg_std)

# save isolation forest model
with open(f'iforest_{this_run_id}.p', 'wb') as f:
    pickle.dump(iforest, f)

# TODO: sort this out
"""
from plotnine import *
% matplotlib
inline
(ggplot(df_ia_agg_scored, aes('risk_score')) + geom_histogram(bins=100))
​"""

# define outliers as having risk score > 0.8
outliers = df_ia_agg_scored[df_ia_agg_scored['risk_score'] >0.8]
len(outliers)

# looking at the IAs in descending order of risk score
outliers_sorted = outliers.sort_values('risk_score',ascending=False)
outliers_sorted.head()

# fit explainer model (not currently used)
feat_names = list(df_ia_agg_std.columns)
import lime
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(df_ia_agg_std.as_matrix(),
                                                   feature_names=feat_names,
                                                   class_names=['score'],
                                                   verbose=False,
                                                   mode='regression')

# example explainer instance
exp = explainer.explain_instance(df_ia_agg_scored.loc[outliers.sort_values('score',ascending=False).index[0],:'Clients_With_More_Than_One_KYC_Change'].as_matrix(),iforest.decision_function,num_features=10)

# example explainer features with negative contributions to anomaly score
[r for r in exp.as_list() if r[1] < 0]

# example explainer features in notebook format
exp.show_in_notebook(show_table=True)

def val_quartile(x):
    '''
    Return bin for min-max quartile value
    '''
    return ceil(x*4)

def minmax_quartiles(x):
    '''
    Get min-max quartiles for column
    '''
    minmax = (df_ia_agg.iloc[:,0].values - df_ia_agg.iloc[:,0].min()) / (df_ia_agg.iloc[:,0].max()-df_ia_agg.iloc[:,0].min())
    return x.apply(val_quartile)

# get explainer results for each instance
# DBG Load from pickle to avoid this step, which takes about 1.5 hours to run on a laptop. 
# df_ia_agg_scored['explainer_results'] = df_ia_agg_std.apply(lambda x: explainer.explain_instance(x.values,iforest.decision_function,num_features=10),axis=1)
# pickle.dump(df_ia_agg_scored, open(PICKLED_DIR + '/df_ia_agg_scored_explainer_pipeline.pkl', 'wb'))
df_ia_agg_scored = pickle.load(open(PICKLED_DIR + '/df_ia_agg_scored_explainer.pkl', 'rb'))

# list of explainer results (not currently used)
df_ia_agg_scored['exp_list'] = df_ia_agg_scored['explainer_results'].apply(lambda x: x.as_list())

# list of KRIs sorted by number of standard deviations from mean
df_ia_agg_scored['largest_list'] = df_ia_agg_scored.iloc[:,:-4].apply(lambda x: list(df_ia_agg_scored.columns[x.values.argsort()[-5:][::-1]]),axis=1)

## Dumping pre-generated dataframes

import pickle

with open(f'df_ia_agg_{this_run_id}.p','wb') as f:
    pickle.dump(df_ia_agg_scored,f)

# TEST
print('Pickling after fit')
pickle.dump(df_ia_agg_scored, open(PICKLED_DIR + '/df_ia_agg_scored.pkl', 'wb'))

################################### HITLIST ###################################

## Generating data for GUI

## Importing previously created dataframe

# DBG
#import pickle
#with open('df_ia_agg_08082018.p','rb') as f:
#    df_ia_agg_scored = pickle.load(f)
with open(PICKLED_DIR + '/df_ia_agg_scored.pkl','rb') as f:
    df_ia_agg_scored = pickle.load(f)

# workaround since this column was not working correctly
del df_ia_agg_scored['Flagged_Trades in Restricted list _AllTime']
def remcol(x):
    newl = []
    for el in x:

        if not 'Flagged_Trades in Restricted list _AllTime' in x:
            newl.append(el)
    return newl
df_ia_agg_scored['exp_list'] = df_ia_agg_scored['exp_list'].apply(remcol)

## Creating hit list

import random, string

# DBG random
random.seed(123)

# create new dataframe with unique IDs for each IA along with their name and risk scores
hit_list = pd.DataFrame(index=[i for i in range(len(df_ia_agg_scored))])

# create unique three alphanumeric character ids
idset = []
for i in range(len(hit_list)):
    while True:
        thisitm = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
        if thisitm not in idset:
            idset.append(thisitm)
            break

# add ids, names, risk scores into dataframe
hit_list['IA ID'] = idset
hit_list['IA Name'] = df_ia_agg_scored.index
hit_list['IA Risk Score'] = df_ia_agg_scored['risk_score'].values

def getPeriod(x):

    '''
    get feature names and coefficients from explainer text
    '''

    l = []
    coefs = []
    for feat_desc, coef in x:

        if '>' in feat_desc:
            feat_name = feat_desc.split('>')[0]
        elif '<' in feat_desc:
            feat_name = feat_desc.split('<')[0]
        else:
            feat_name = feat_desc.split('=')[0]
        feat_name = feat_name.strip()
        l.append(feat_name)
        coefs.append(coef)

    return l, coefs

def getTerm(x):
    '''
    Get the short, medium, long term risk based on proportion of TMs found in explainer model list
    '''
    tm_cnt = 0
    tot_cnt = 0
    for el in x:

        if 'TM' in el:
            tm_cnt += 1
        tot_cnt += 1
    if tm_cnt / tot_cnt < (1 / 3):
        return 'L'
    elif tm_cnt / tot_cnt < (2 / 3):
        return 'M'
    else:
        return 'S'

hit_list['S/M/L term'] = df_ia_agg_scored['exp_list'].apply(lambda x: getPeriod(x)[0]).apply(getTerm).values
hit_list.to_excel(PICKLED_DIR + '/' + f'Hit_List_{this_run_id}.xls')
# DBG
# hit_list = pd.read_excel('Hit_List.xls')
hit_list = pd.read_excel(PICKLED_DIR + '/' + f'Hit_List_{this_run_id}.xls')

# TEST
print('Pickling after hitlist')
pickle.dump(hit_list, open(PICKLED_DIR + '/hit_list.pkl', 'wb'))
pickle.dump(df_ia_agg_scored, open(PICKLED_DIR + '/df_ia_agg_scored.2.pkl', 'wb'))

################################### HITLISTEXP ###################################

# DBG random
np.random.seed(123)

def sensible_decrease(x):
    '''
    Function to create increase/decrease values in risk score that are reasonable (i.e don't imply a
    risk score greater than 100 or less than 0)
    '''
    if 100-x < 10/100:
        return np.random.randint(x-100,10)/100
    elif x<10/100:
        return np.random.randint(-10,x)/100
    else:
        return np.random.randint(-10,10)/100

def los_transform(x):
    '''
    Convert LOS from text/integer combination into integers only
    '''
    if x == 'OVER 10':
        return 10
    elif 'F' in x:
        return 0
    else:
        return int(x.strip())

def personal_less_than_client(x):
    '''
    Correction for Personal ROA to be less than client ROA
    '''
    rand_val= np.random.random()*(0.25)+0.
    if x['Client ROA(%)'] < rand_val:
        return max(x['Client ROA(%)']-0.05,0)
    else:
        return rand_val

# bridge table between IA and region
ia_region_bridge = gross_revs[['FLATTEN_NAME','REGION']].drop_duplicates(['FLATTEN_NAME'])

## Creating the expanded hit list

# calculate annual gross revenues
revs_annual = gross_revs.groupby('FLATTEN_NAME')['REV_CURMONTH'].sum().reset_index()
revs_annual.columns = ['IA Name','REV_ANNUAL']
# take original hit list and add in branch number and name
hit_list_expanded = pd.merge(left=hit_list,right=acct_trades.drop_duplicates('IA_NAME')[['IA_NAME','RR_BRANCH_NUM','WM_PHY_BRANCH_NAME']],left_on='IA Name',right_on='IA_NAME')

# add increase/decreaes to risk score
hit_list_expanded['Increase/Decrease in Risk score(delta)'] = hit_list_expanded['IA Risk Score'].apply(sensible_decrease)

# add region information
hit_list_expanded = pd.merge(left=hit_list_expanded,right=ia_region_bridge,left_on='IA Name',right_on='FLATTEN_NAME')

# add gross revenues
hit_list_expanded = pd.merge(left=hit_list_expanded,right=revs_annual,on='IA Name',how='inner')

# add minmax scaled gross revenues
hit_list_expanded['REV_MinMax'] = (hit_list_expanded['REV_ANNUAL'] - hit_list_expanded['REV_ANNUAL'].min())/(hit_list_expanded['REV_ANNUAL'].max()-hit_list_expanded['REV_ANNUAL'].min())

# calculate AUM using linear interpolation of gross revenues
hit_list_expanded['AUM(in million $)'] = hit_list_expanded['REV_MinMax']*(1100-30)+30

# calculate AUM growth from abs of normal distribution w mean 10 std 10
hit_list_expanded['AUM growth %'] = hit_list_expanded['AUM(in million $)'].apply(lambda x: 0.10*abs(np.random.normal(0,1))*100)

# get region rank
hit_list_expanded['Rank #'] = hit_list_expanded.groupby(['REGION'])['REV_MinMax'].rank(ascending=False).astype(int).values

# get number of accounts as proxy for households
num_accts = acct_trades.groupby(['IA_NAME','ACCT_ID'])['TRD_TRADE_ID'].count().reset_index().groupby('IA_NAME')['TRD_TRADE_ID'].count().reset_index()
num_accts.columns = ['IA_NAME','Num_Accts']
hit_list_expanded = pd.merge(hit_list_expanded,num_accts,left_on='IA Name',right_on='IA_NAME',how='inner')
hit_list_expanded['Households'] = hit_list_expanded['Num_Accts']

# get number of IAs within region
region_ia_counts = hit_list_expanded.groupby('REGION')['IA Name'].count().reset_index()
region_ia_counts.columns = ['REGION','Total # of IAs within region']
hit_list_expanded = pd.merge(hit_list_expanded,region_ia_counts,on='REGION',how='inner')

# get LOS
gross_revs['LOS(yrs)'] = gross_revs['LOS'].apply(los_transform)
hit_list_expanded = pd.merge(hit_list_expanded,gross_revs[['FLATTEN_NAME','LOS(yrs)']].drop_duplicates('FLATTEN_NAME'),left_on='IA Name',right_on='FLATTEN_NAME')

# get average number of trades
avg_trades = acct_trades.groupby(['IA_NAME','TRD_MONTH'])['TRD_TRADE_ID'].count().reset_index().groupby('IA_NAME').mean().reset_index()
avg_trades.columns = ['IA Name','Avg Trades(/month)']
hit_list_expanded = pd.merge(hit_list_expanded,avg_trades,on='IA Name',how='left')

# generate ROA and pro accounts values using random numbers, margin = 0
hit_list_expanded['Client ROA(%)'] = hit_list_expanded['IA Name'].apply(lambda x: np.random.random()*(1.75)+0.)
hit_list_expanded['Pro Acct(in million $)'] = hit_list_expanded['IA Name'].apply(lambda x: min(5,max(0.2,np.random.normal(0.5,1.))))
hit_list_expanded['Margin(in $K)'] = 0
hit_list_expanded['Personal ROA (in %)'] = hit_list_expanded['IA Name'].apply(lambda x: np.random.random()*(1.75)+0.)

# add concatenation of branch number and name
hit_list_expanded['Branch #'] = hit_list_expanded['RR_BRANCH_NUM'].astype(str) + '-' + hit_list_expanded['WM_PHY_BRANCH_NAME']

# apply personal ROA correction to be less than client ROA
hit_list_expanded['Personal ROA (in %)'] = hit_list_expanded.apply(personal_less_than_client,axis=1)

## Output desired columns for hit list and expanded hit list to excel

hit_list_cols = ['IA ID','IA Name','IA Risk Score','S/M/L term','Increase/Decrease in Risk score(delta)',
                'AUM(in million $)','AUM growth %']
hit_list_out = hit_list_expanded[hit_list_cols]
hit_list_out.to_excel(PICKLED_DIR + '/' + f'hit_list_{this_run_id}.xlsx')

hit_list_expanded_cols = ['IA ID','Rank #','Total # of IAs within region',
                          'Households','LOS(yrs)','Avg Trades(/month)','Client ROA(%)',
                          'Pro Acct(in million $)', 'Margin(in $K)', 'Personal ROA (in %)',
                          'Branch #']
hit_list_expanded_out = hit_list_expanded[hit_list_expanded_cols]
hit_list_expanded_out.to_excel(PICKLED_DIR + '/' + f'hit_list_expanded_{this_run_id}.xlsx')

# TEST
print('Pickling after hitlistexp')
pickle.dump(hit_list_out, open(PICKLED_DIR + '/hit_list_out.pkl', 'wb'))
pickle.dump(hit_list_expanded, open(PICKLED_DIR + '/hit_list_expanded.pkl', 'wb'))
pickle.dump(hit_list_expanded_out, open(PICKLED_DIR + '/hit_list_expanded_out.pkl', 'wb'))

################################### KRIDETAILS ###################################

# load isolation forest model if needed
import pickle
# DBG
#with open('iforest.p','rb') as f:
with open('iforest_02.p','rb') as f:
    iforest = pickle.load(f)

## KRI details

# create dictionary of feature names to features names to be used in GUI
gui_kri_names = pd.read_excel(f'GUI KRI Naming.xlsx')
gui_kri_names.index = gui_kri_names['Features']
del gui_kri_names['Features']
gui_name_dict = gui_kri_names.to_dict()['Name for GUI']

# get list of all used KRIs and count
all_feats = []
all_cnt = 0
for v in df_ia_agg_scored['largest_list']:
    for el in v:
        if 'Flagged_Trades in Restricted list _AllTime' not in el:
            all_cnt += 1
            if el not in all_feats:
                all_feats.append(el)

# create dataframe for KRIs
kri_details = pd.DataFrame(index=list(range(len(all_feats))))
kri_details['KRI ID'] = kri_details.index
kri_details['KRI Description'] = all_feats
kri_details['KRI Description'] = kri_details['KRI Description'].map(gui_name_dict)

# export kri details
kri_details.to_excel(PICKLED_DIR + '/' + f'kri_details_{this_run_id}.xlsx',index=False)

## IA KRI Mapping

# get mapping of IAs to KRIs
all_coefs = []
all_feats = []
all_quartiles = []
all_ia = []
feats_rolled = df_ia_agg_scored['largest_list']
for i, feats in enumerate(feats_rolled):

    for f in feats:

        if 'Flagged_Trades in Restricted list _AllTime' not in f:
            # append feature name
            all_feats.append(f)

            # append minmax quartile value
            all_quartiles.append(minmax_quartiles(df_ia_agg_scored[f])[i])

            # append IA name
            all_ia.append(list(df_ia_agg_scored.index)[i])

ia_kri_mapping = pd.DataFrame({'IA Name': all_ia,
                               'KRI Description': all_feats,
                               'Quartile value': all_quartiles})
ia_kri_mapping['KRI Description'] = ia_kri_mapping['KRI Description'].map(gui_name_dict)

# append IA ID to mapping table
ia_kri_mapping = pd.merge(ia_kri_mapping,hit_list_expanded[['IA Name','IA ID']],on='IA Name',how='inner')

# append kri ID to mapping table
ia_kri_mapping = pd.merge(ia_kri_mapping,kri_details,on='KRI Description',how='inner')

# output IA KRI mapping table to excel
ia_kri_mapping_cols = ['IA ID','KRI ID','Quartile value']
ia_kri_mapping_out = ia_kri_mapping[ia_kri_mapping_cols].sort_values(['IA ID','KRI ID'])
ia_kri_mapping_out.to_excel(PICKLED_DIR + '/' + f'ia_kri_mapping_{this_run_id}.xlsx',index=False)

# TEST
print('Pickling after kridetails')
pickle.dump(kri_details, open(PICKLED_DIR + '/kri_details.pkl', 'wb'))
pickle.dump(ia_kri_mapping, open(PICKLED_DIR + '/ia_kri_mapping.pkl', 'wb'))
pickle.dump(ia_kri_mapping_out, open(PICKLED_DIR + '/ia_kri_mapping_out.pkl', 'wb'))

################################### WELLBEING ###################################

# DBG random
np.random.seed(123)

## Wellbeing

# create template columns for IA behaviour
ia_repeated = np.repeat(hit_list['IA ID'].values,12)
months_repeated = np.tile([5,5,5,6,6,6,7,7,7,8,8,8],len(hit_list['IA ID'].values))
desc_repeated = np.tile(['anxiety','fulfillment','judgement'],int(len(ia_repeated)/3))

# generate anxiety, fulfilment, and judgement scores
a = []
for i in range(len(hit_list['IA ID']) * 4):
    if i % 4 == 0:
        # start new anxiety score after four months
        a.append(np.random.randint(10, 80, 1)[0])
    else:
        # create a new anxiety score and random amount above or below the previous score
        a.append(int(min(80, max(10, a[i - 1] + [-1, 1][np.random.choice(2)] * int(3 * np.random.random(1))))))

# create anxiety, fulfilment, and judgement numpy arrays
a = np.array(a)
f = 100 - a + 20
j = 100 - a + 10

# correct instances where above 100 to be equal to 100
new_arr = []
for i in range(len(a)):

    for k in range(3):

        if k == 0:
            new_arr.append(a[i])
        elif k == 1:
            if f[i] > 100:
                new_arr.append(100)
            else:
                new_arr.append(f[i])
        else:
            if j[i] > 100:
                new_arr.append(100)
            else:
                new_arr.append(j[i])

# test plotting of scores for small section of values
# TODO install plotnine
#from plotnine import *
df_test = pd.DataFrame({'a':a,'j':j,'f':f,'i':[i for i in range(len(a))]})
# TODO install plotnine
#(ggplot(data=df_test) + geom_line(aes('i','a')) + geom_line(aes('i','f')) + geom_line(aes('i','j')) + scale_x_continuous(limits=[8,16]))

# output wellbeing values to excel
wellbeing = pd.DataFrame({'IA ID':ia_repeated,
                          'Month': months_repeated,
                          'Metric description': desc_repeated,
                          'Value':new_arr})
wellbeing.to_excel(PICKLED_DIR + '/' + f'wellbeing_{this_run_id}.xlsx', index=False)

# TEST
print('Pickling after wellbeing')
pickle.dump(wellbeing, open(PICKLED_DIR + '/wellbeing.pkl', 'wb'))

################################### BRANCHBIN ###################################

## Branch Binned

def risk_bin(x):
    '''
    Function to output binned risk as high, medium, or low based on risk score
    '''
    if x > 0.8:
        return 'H'
    elif x > 0.5:
        return 'M'
    else:
        return 'L'

# bin by branch and output to excel
hit_list_expanded['risk_bin'] = hit_list_expanded['IA Risk Score'].apply(risk_bin)
branch_binned = hit_list_expanded.groupby(['Branch #', 'risk_bin'])['IA ID'].count().reset_index()
branch_binned[branch_binned['risk_bin'] == 'H']
branch_binned.to_excel(PICKLED_DIR + '/' + f'branch_binned_{this_run_id}.xlsx', index=False)

## Joined tables
joined_tables = pd.merge(hit_list_expanded,
    ia_kri_mapping,
    left_on='IA ID',
    right_on='IA ID',how='left').to_excel(PICKLED_DIR + '/' + f'joined_tables_{this_run_id}.xlsx',index=False)

# TEST
print('Pickling after branchbin')
pickle.dump(branch_binned, open(PICKLED_DIR + '/branch_binned.pkl', 'wb'))
pickle.dump(joined_tables, open(PICKLED_DIR + '/joined_tables.pkl', 'wb'))
