import pandas as pd
import pickle
import random
import string
from pathlib import Path

from traice import batchstep
from traice import utils

class HitList(batchstep.BatchStep):

    def __init__(self, input_dir, pickled_dir, out_path,cred_dict):

        super().__init__(input_dir, pickled_dir, out_path,cred_dict)

        # Inputs
        self.df_ia_agg_scored = None

        # Outputs
        self.hit_list = None

    def remcol(self, x):

        newl = []
        for el in x:

            if not 'Flagged_Trades in Restricted list _AllTime' in x:
                newl.append(el)
        return newl

    def getPeriod(self, x):

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

    def getTerm(self, x):

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

    ## Creating hit list
    def run_step(self):
        
        random.seed(123)

        self.df_ia_agg_scored = pickle.load(open(self.pickled_dir + '/df_ia_agg_scored.pkl', 'rb'))
        print('begin')
        # DBG
        #with open('df_ia_agg_02.p','rb') as f:
        #    df_ia_agg_scored = pickle.load(f)

        # workaround since this column was not working correctly
        #del self.df_ia_agg_scored['Flagged_Trades in Restricted list _AllTime']
        self.df_ia_agg_scored['exp_list'] = self.df_ia_agg_scored['exp_list'].apply(self.remcol)

        # create new dataframe with unique IDs for each IA along with their name and risk scores
        self.hit_list = pd.DataFrame(index=[i for i in range(len(self.df_ia_agg_scored))])

        # create unique three alphanumeric character ids
        idset = []
        for i in range(len(self.hit_list)):
            while True:
                thisitm = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
                if thisitm not in idset:
                    idset.append(thisitm)
                    break

        # add ids, names, risk scores into dataframe
        self.hit_list['IA ID'] = idset
        self.hit_list['IA Name'] = self.df_ia_agg_scored.index
        self.hit_list['IA Risk Score'] = self.df_ia_agg_scored['risk_score'].values

        self.hit_list['S/M/L term'] = self.df_ia_agg_scored['exp_list'].apply(lambda x: self.getPeriod(x)[0]).apply(self.getTerm).values
        #self.hit_list.to_excel(f'Hit_List_{this_run_id}.xls')
        import pymysql
        import mysql.connector
        from sqlalchemy import create_engine
        engine = create_engine("mysql+pymysql://" + self.cred_dict['username'] + ":" + self.cred_dict['password'] + "@" + "localhost" + "/" + "traice2")

        self.hit_list.to_sql('hit_list', con = engine, if_exists = 'replace',index = False, chunksize = 1000)
        
        #self.hit_list.to_excel(self.pickled_dir + '/Hit_List_02.xls')
        #self.hit_list = pd.read_excel(self.pickled_dir + '/Hit_List_02.xls')

        # We need to re-pickle df_ia_agg_scored since it was changed
        pickle.dump(self.hit_list, open(self.pickled_dir + '/hit_list.pkl', 'wb'))
        pickle.dump(self.df_ia_agg_scored, open(self.pickled_dir + '/df_ia_agg_scored.2.pkl', 'wb'))
