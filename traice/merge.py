import pandas as pd
import pickle
from pathlib import Path

from traice import batchstep
from traice import utils

class Merge(batchstep.BatchStep):

    def __init__(self, input_dir, pickled_dir, out_path,cred_dict):

        super().__init__(input_dir, pickled_dir, out_path,cred_dict)

        # Input tables. These will be extracted from pickle files
        self.trades = None              # all trades
        self.acct = None                # accounts       
        self.trd_ind = None             # trade indicators
        self.acct_trade_bridge = None   # table bridging accounts and trades tables

        # Outputs. These will be pickled        
        self.acct_trades = None         
        self.flagged_extended = None
        self.dummy_cols = None                        
        self.code_dict = None                              

    def merge_tables(self):

        # merging the account and trades tables
        acct_bridge = pd.merge(self.acct, 
            self.acct_trade_bridge,
            left_on=['BIZ_DATE', 'ACCT_ID'],
            right_on=['ACCT_DATE', 'ACCT_ID'],
            how='inner')
        
        self.acct_trades = pd.merge(acct_bridge,
            self.trades,
            on=['ACCT_ID', 'TRD_BIZ_DATE'],
            how='inner')

        # Adding month column to account trades
        self.acct_trades['TRD_MONTH'] = self.acct_trades['TRD_BIZ_DATE'].astype('str').apply(
            lambda x: x[:x.index('/') + 1]+x[x.rfind('/')+1:])

        
        print(self.acct_trades['TRD_MONTH'])
    # Convert the categorical trade surveillance indicators into one column for each,
    # to make the data suitable for training in a model.
    def convert_categorical(self):

        # reduce columns to consume less memory
        keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 
            'TRD_MONTH', 'TRADE_INDICATOR_ID_DESC', 
            'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION','IDENTIFIER_TYPE','ORDER_TYPE','SETTLEMENT_CURRENCY']
        
        print(self.acct_trades.columns)
        print(self.trd_ind.columns)
        df_all_flags_merge = pd.merge(
            left=self.acct_trades, right=self.trd_ind, on='TRD_TRADE_ID', how='left')[keep_cols]
              
      
        # convert flags/categories to one flag per column
        dummies = pd.get_dummies(df_all_flags_merge['TRADE_INDICATOR_ID_DESC'])
        self.dummy_cols = dummies.columns
        print('dummmiiiiiiies',dummies)
        # merge table and broken out flag columns
        df_all_flags_dummies_merge = pd.merge(
            df_all_flags_merge, dummies, left_index=True, right_index=True, how='left')

        # DBG
        #keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'WM_PHYSICAL_BRANCH_ID']
        keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION','IDENTIFIER_TYPE','ORDER_TYPE','SETTLEMENT_CURRENCY']
        self.flagged_extended = df_all_flags_dummies_merge.groupby(keep_cols)[self.dummy_cols].sum().reset_index()
        print('flagged_extended',self.flagged_extended)

    def store_indicator_info(self):

        # read in descriptions for the various flags, store in dictionary
        trade_indicator_path = None
        #for f in self.input_files:
         #   print(f)
          #  if f.endswith('trade indicator.xlsx'):
           #     trade_indicator_path = f
        #print("trade_indicator_path is",trade_indicator_path)
        temp_df=self.trade_indicator
        temp_df = temp_df.iloc[:-1, :][['TRADE_INDICATOR_DESC', 'Short_Description']]
        temp_df.index = temp_df['TRADE_INDICATOR_DESC']
        del temp_df['TRADE_INDICATOR_DESC']
        self.code_dict = temp_df.to_dict()['Short_Description']

    def run_step(self):

        self.trades = pickle.load(open(self.pickled_dir + '/trades.pkl', 'rb'))
        self.acct = pickle.load(open(self.pickled_dir + '/acct.pkl', 'rb'))     
        self.trd_ind = pickle.load(open(self.pickled_dir + '/trd_ind.pkl', 'rb'))
        self.acct_trade_bridge = pickle.load(open(self.pickled_dir + '/acct_trade_bridge.pkl', 'rb'))
        self.trade_indicator = pickle.load(open(self.pickled_dir + '/trade_indicator.pkl', 'rb'))


        # merge tables using joins
        self.merge_tables()

        # convert categorical trade surveillance indicator values to dummy values
        self.convert_categorical()

        # create a dictionary of indicator values and descriptions
        self.store_indicator_info()
        
        print('doooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooont',self.acct_trades.columns)                
        pickle.dump(self.acct_trades, open(self.pickled_dir + '/acct_trades.pkl', 'wb'))
        pickle.dump(self.flagged_extended, open(self.pickled_dir + '/flagged_extended.pkl', 'wb'))
        pickle.dump(self.dummy_cols, open(self.pickled_dir + '/dummy_cols.pkl', 'wb'))
        pickle.dump(self.code_dict, open(self.pickled_dir + '/code_dict.pkl', 'wb'))



    
