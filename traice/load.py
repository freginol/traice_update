import pandas as pd
import pickle
import os
import mariadb
import sys
import pandas as pd
from pathlib import Path

from traice import batchstep

TABLE_PICKLE_MAPPING = {
    'lex_trade' : 'trades.pkl',
    'lex_acct_trade_bridge' : 'acct_trade_bridge.pkl',
    'lex_acct' : 'acct.pkl',
    'LEX_TRD_IND' : 'trd_ind.pkl',
    'lex_complaints' : 'complaints.pkl',
    'lex_gross_ia_income' : 'gross_revs.pkl',
    'GUI_KRI_NAMING' : 'gui_kri_naming.pkl',
    'trade_indicator': 'trade_indicator.pkl',
    'emails': 'emails.pkl',
    'aml':'aml.pkl'
}
input_files = [
    'GUI_KRI_NAMING',
    'lex_acct',
    'lex_acct_trade_bridge',
    'lex_complaints',
    'lex_gross_ia_income',
    'lex_trade',
    'LEX_TRD_IND',
    'trade_indicator',
    'emails',
    'aml'
]

class Load(batchstep.BatchStep):

    def __init__(self, input_files, pickled_dir, completion_file,cred_dict):
        super().__init__(input_files, pickled_dir, completion_file,cred_dict)
    
    def fetch_data(self,table_name):
        print('burger',self.cred_dict)
        # Connect to MariaDB Platform
        try:
            conn = mariadb.connect(
                user=self.cred_dict['username'],
                password=self.cred_dict['password'],
                host="127.0.0.1",
                port=3306,
                database="traice2"

            )
        except mariadb.Error as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            sys.exit(1)
        print('connection done')
        # Get Cursor
        cur = conn.cursor()
        cur.execute('select * from %s;'%table_name)
        my_list = []
        for row in cur:
            my_list.append(row)
        
        df=pd.DataFrame(my_list)
        cur.execute('show columns from %s;'%table_name)
        column_list=[]
        for each_column in cur:
            column_list.append(each_column[0])
        df.columns=column_list
        return df
        

    def run_step(self):
        
        pickle_dict_out = {}
        tables = []
        tables_idx = {}
        print(self.cred_dict)
        for file_name in input_files:
            df = self.fetch_data(file_name)
            print(file_name,df)
            file_basename = file_name
            pickle_path = self.pickled_dir + '/' + TABLE_PICKLE_MAPPING[file_basename]
            pickle.dump(df, open(pickle_path, 'wb'))
    