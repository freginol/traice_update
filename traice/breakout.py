import pandas as pd
import pickle
import datetime
import hashlib
from pathlib import Path
import unicodedata
from traice import batchstep
import re

class BreakOut(batchstep.BatchStep):

    def __init__(self, input_dir, pickled_dir, out_path,cred_dict):

        super().__init__(input_dir, pickled_dir, out_path,cred_dict)

        # Inputs
        self.acct_trades = None
        self.complaints = None

        # Outputs
        self.ia_branch_region_bridge = None
        self.pro_trades = None
        self.cancelled_trades = None
        self.reversals = None
        self.complaints = None
        self.acct_kyc_12m = None
        self.traded_under_other_ia = None

    def get_pro_trades(self):

        keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'PRO_ACCOUNT', 'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION','IDENTIFIER_TYPE','ORDER_TYPE','SETTLEMENT_CURRENCY']
        self.pro_trades = self.acct_trades[self.acct_trades['PRO_ACCOUNT'] == 'PRO'][keep_cols]
    

    def get_cancelled(self):

        keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'CANCEL_INDICATOR', 'WM_PHYSICAL_BRANCH_ID',
             'WM_PHY_BRANCH_REGION','IDENTIFIER_TYPE','ORDER_TYPE','SETTLEMENT_CURRENCY']
        self.cancelled_trades = self.acct_trades[self.acct_trades['CANCEL_INDICATOR'] == 'CXL'][keep_cols]

    def get_reversals(self):

        # calculate where net buy / sell quantity == 0
        self.ia_branch_region_bridge = self.acct_trades[['IA_NAME', 'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION']].drop_duplicates('IA_NAME')
        self.acct_trades['Signed_Quantity'] = self.acct_trades['BUY_SELL_INDICATOR'].apply(lambda x: 1 if x == 'B' else -1) * self.acct_trades['QUANTITY']
        trd_grp2 = self.acct_trades.groupby(['TRD_BIZ_DATE', 'TRD_MONTH', 'IA_NAME', 'ACCT_ID', 'SEC_SECURITY_ID'])['Signed_Quantity'].sum().reset_index()
        self.reversals = trd_grp2[trd_grp2['Signed_Quantity'] == 0].groupby(['TRD_BIZ_DATE', 'TRD_MONTH', 'IA_NAME'])['Signed_Quantity'].count().reset_index()
        self.reversals['TRD_TRADE_ID'] = self.reversals['Signed_Quantity']
        self.reversals = pd.merge(self.reversals, self.ia_branch_region_bridge, on='IA_NAME', how='inner')


    def get_complaints(self):

        # convert month to same format as other month columns
        #print(self.complaints['DATE_RECEIVED'])
        
        self.complaints['TRD_MONTH'] = self.complaints['DATE_RECEIVED'].apply(
            lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').strftime('%b-%y').upper())
        
        self.complaints['IA_NAME']=self.complaints['IA_NAME'].apply(lambda x: "".join(re.split("[^a-zA-Z ]*", x)))
        #print('this is a complaintttttttttt..........',self.complaints)
        #for i in self.complaints['IA_NAME']:
        #    print(i,len(i))
        #print('next one')
        #for i in self.ia_branch_region_bridge['IA_NAME']:
        #    print(i.strip(),len(i.strip()))
        self.ia_branch_region_bridge['IA_NAME']=self.ia_branch_region_bridge['IA_NAME'].apply(lambda x: x.strip())
        #self.complaints.to_excel(self.pickled_dir + '/complaintstrial.xls')
        #print(self.complaints['IA_NAME'],'fine',self.ia_branch_region_bridge['IA_NAME'])
        self.complaints = pd.merge(self.complaints, self.ia_branch_region_bridge, on='IA_NAME', how='inner')
        
        #print('this is a ia branch region bridge..........',self.ia_branch_region_bridge)
        self.complaints['TRD_TRADE_ID'] = self.complaints.index
        #print('merged',self.complaints)

       
        #self.ia_branch_region_bridge.to_excel(self.pickled_dir + '/iabranchregiontrial.xls')
        
        #exit()


    def get_kyc_changes(self):

        # calculate number of kyc changes by taking a concatenation of all of the KYC columns and counting the number
        # of distinct entries per account, then rolling up to determine the number of clients by IA with more than
        # two changes
        kyc_cols = [col for col in self.acct.columns if col[:3] == 'KYC']
        kyc_only = self.acct_trades[kyc_cols]
        #print('colsssss',kyc_only)
        self.acct_trades['KYC_Hash'] = kyc_only.astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(str(x).encode('ascii')).hexdigest())
        #print("output",self.acct_trades['KYC_Hash'])
        self.acct_kyc_12m = self.acct_trades.groupby(['ACCT_ID', 'IA_NAME', 'KYC_Hash'])['BIZ_DATE'].count().reset_index().groupby(['ACCT_ID', 'IA_NAME'])['KYC_Hash'].count() - 1
        self.acct_kyc_12m = self.acct_kyc_12m[self.acct_kyc_12m > 2].reset_index().groupby('IA_NAME')['KYC_Hash'].count().reset_index()
    
    def get_traded_under_different_ia(self):

        keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION','IDENTIFIER_TYPE','ORDER_TYPE','SETTLEMENT_CURRENCY']
        print(self.acct_trades.columns)
        self.traded_under_other_ia = self.acct_trades[self.acct_trades['IA_NAME'] != self.acct_trades['TRADE_IA_NAME']][keep_cols]

  # def get_order_type(self):
   #     keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'WM_PHYSICAL_BRANCH_ID', 'WM_PHY_BRANCH_REGION','IDENTIFIER_TYPE','SETTLEMENT_CURRENCY']
    #    self.order_type=self.acct_trades.groupby(keep_cols)['ORDER_TYPE'].count().reset_index()
     #   print('get..',(self.order_type[['IA_NAME','ORDER_TYPE']]))
      #  print('get..',len(self.order_type[['IA_NAME','ORDER_TYPE']]))'''

    def get_order_type_MARKET(self):
        keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'CANCEL_INDICATOR', 'WM_PHYSICAL_BRANCH_ID','WM_PHY_BRANCH_REGION','IDENTIFIER_TYPE','ORDER_TYPE','SETTLEMENT_CURRENCY']
        self.order_type_MARKET = self.acct_trades[self.acct_trades['ORDER_TYPE'] == 'MARKET'][keep_cols]
    
    
    def get_order_type_LIMIT(self):
        keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'CANCEL_INDICATOR', 'WM_PHYSICAL_BRANCH_ID','WM_PHY_BRANCH_REGION','IDENTIFIER_TYPE','ORDER_TYPE','SETTLEMENT_CURRENCY']
        self.order_type_LIMIT = self.acct_trades[self.acct_trades['ORDER_TYPE'] == 'LIMIT'][keep_cols]

    def get_order_type_STOP(self):
        keep_cols = ['TRD_TRADE_ID', 'IA_NAME', 'TRD_MONTH', 'CANCEL_INDICATOR', 'WM_PHYSICAL_BRANCH_ID','WM_PHY_BRANCH_REGION','IDENTIFIER_TYPE','ORDER_TYPE','SETTLEMENT_CURRENCY']
        self.order_type_STOP = self.acct_trades[self.acct_trades['ORDER_TYPE'] == 'STOP'][keep_cols]

    # Break out each of the following indicators:
    # - Pro trades: IA traded on their own account
    # - Cancelled: Cancelled trades
    # - Reversals: Taking a buy position and sell position that together net out to 0, 
    #   indicating needless trading
    # - Complaints: customer complaints
    # - KYC Changes: Know Your Customer changes in the last 12 months
    # - Trading under different IA: IA traded under another IA's name (not always an
    #   indicator of questionable behavior) 
    def run_step(self):
        
        self.acct = pickle.load(open(self.pickled_dir + '/acct.pkl', 'rb'))
        self.acct_trades = pickle.load(open(self.pickled_dir + '/acct_trades.pkl', 'rb'))
        self.complaints = pickle.load(open(self.pickled_dir + '/complaints.pkl', 'rb'))    
        self.get_pro_trades()
        self.get_cancelled()
        self.get_reversals()
        self.get_complaints()
        self.get_kyc_changes()
        self.get_traded_under_different_ia()
        self.get_order_type_MARKET()
        self.get_order_type_LIMIT()
        self.get_order_type_STOP()
        print('length trades',len(self.pro_trades),len(self.cancelled_trades),len(self.reversals),len(self.acct_kyc_12m),len(self.traded_under_other_ia))

        # acct_trades was changed by get_reversals() and get_kyc_changes() and needs to be saved again
        pickle.dump(self.acct_trades, open(self.pickled_dir + '/acct_trades.2.pkl', 'wb'))
        # complaints was changed by get_complaints() and needs to be saved again
        pickle.dump(self.complaints, open(self.pickled_dir + '/complaints.2.pkl', 'wb'))
        pickle.dump(self.ia_branch_region_bridge, open(self.pickled_dir + '/ia_branch_region_bridge.pkl', 'wb'))
        pickle.dump(self.pro_trades, open(self.pickled_dir + '/pro_trades.pkl', 'wb'))
        pickle.dump(self.cancelled_trades, open(self.pickled_dir + '/cancelled_trades.pkl', 'wb'))
        pickle.dump(self.reversals, open(self.pickled_dir + '/reversals.pkl', 'wb'))
        pickle.dump(self.acct_kyc_12m, open(self.pickled_dir + '/acct_kyc_12m.pkl', 'wb'))
        pickle.dump(self.traded_under_other_ia, open(self.pickled_dir + '/traded_under_other_ia.pkl', 'wb'))
        pickle.dump(self.order_type_MARKET, open(self.pickled_dir + '/order_type_MARKET.pkl', 'wb'))
        pickle.dump(self.order_type_LIMIT, open(self.pickled_dir + '/order_type_LIMIT.pkl', 'wb'))
        pickle.dump(self.order_type_STOP, open(self.pickled_dir + '/order_type_STOP.pkl', 'wb'))
