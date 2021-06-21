import pandas as pd
import pickle
from pathlib import Path
from math import ceil
import pickle
from traice import batchstep

class KRIDetails(batchstep.BatchStep):

    def __init__(self, input_dir, pickled_dir, out_path,cred_dict):

        super().__init__(input_dir, pickled_dir, out_path,cred_dict)

        # Inputs
        self.df_ia_agg = None
        self.df_ia_agg_scored = None
        self.hit_list_expanded = None

        # Outputs
        self.ia_kri_mapping = None
        self.ia_kri_mapping_out = None

    def val_quartile(self, x):
        '''
        Return bin for min-max quartile value
        '''
        return ceil(x*4)

    def minmax_quartiles(self, x):
        '''
        Get min-max quartiles for column
        '''
        minmax = (self.df_ia_agg.iloc[:,0].values - self.df_ia_agg.iloc[:,0].min()) / (self.df_ia_agg.iloc[:,0].max()-self.df_ia_agg.iloc[:,0].min())
        return x.apply(self.val_quartile)

    ## KRI details
    # TODO Make sure GUI KRI Naming.xlsx is included in inputs
    def run_step(self):
        print('BEGIN')
        self.df_ia_agg = pickle.load(open(self.pickled_dir + '/df_ia_agg.pkl', 'rb'))
        self.df_ia_agg_scored = pickle.load(open(self.pickled_dir + '/df_ia_agg_scored.pkl', 'rb'))
        self.hit_list_expanded = pickle.load(open(self.pickled_dir + '/hit_list_expanded.pkl', 'rb'))
        
        import pymysql
        import mysql.connector
        from sqlalchemy import create_engine
        engine = create_engine("mysql+pymysql://" + self.cred_dict['username'] + ":" + self.cred_dict['password'] + "@" + "localhost" + "/" + "traice2")
        

        # create dictionary of feature names to features names to be used in GUI
        # DBG
        print('BEGIN')
        gui_kri_naming_path = None
        for f in self.input_files:
            if f.endswith('GUI KRI Naming.xlsx'):
                gui_kri_naming_path = f
                break
        
        # DBG        
        #gui_kri_names = pd.read_excel(f'GUI KRI Naming.xlsx')
        pickled_dir = './csvfiles/pickled/batch'
        gui_kri_names=pickle.load(open(pickled_dir + '/gui_kri_naming.pkl', 'rb'))
        print('df is',gui_kri_names)
        gui_kri_names.index = gui_kri_names['Features']
        del gui_kri_names['Features']
        gui_name_dict = gui_kri_names.to_dict()['Name_for_GUI']
        print('the infamous dictionary',gui_name_dict)

        # get list of all used KRIs and count
        all_feats = []
        all_cnt = 0
        for v in self.df_ia_agg_scored['largest_list']:
            for el in v:
                if 'Flagged_Trades in Restricted list _AllTime' not in el:
                    all_cnt += 1
                    if el not in all_feats:
                        all_feats.append(el)

        print('kri allfeats is ',all_feats)
        # create dataframe for KRIs
        self.kri_details = pd.DataFrame(index=list(range(len(all_feats))))
        self.kri_details['KRI ID'] = self.kri_details.index
        self.kri_details['KRI Description'] = all_feats
        
        self.kri_details['KRI Description'] = self.kri_details['KRI Description'].map(gui_name_dict)
        print('after mapping',self.kri_details)
        print('dictionary is ',gui_name_dict)
        
        # export kri details
        #self.kri_details.to_excel(f'kri_details_{this_run_id}.xlsx',index=False)
        #self.kri_details.to_excel(self.pickled_dir + '/kri_details_02.xlsx',index=False)
        self.kri_details.to_sql('kri_details_02', con = engine, if_exists = 'replace',index = False, chunksize = 1000)

        ## IA KRI Mapping

        # get mapping of IAs to KRIs
        all_coefs = []
        all_feats = []
        all_quartiles = []
        all_ia = []
        feats_rolled = self.df_ia_agg_scored['largest_list']
        for i, feats in enumerate(feats_rolled):

            for f in feats:

                if 'Flagged_Trades in Restricted list _AllTime' not in f:
                    # append feature name
                    all_feats.append(f)

                    # append minmax quartile value
                    all_quartiles.append(self.minmax_quartiles(self.df_ia_agg_scored[f])[i])

                    # append IA name
                    all_ia.append(list(self.df_ia_agg_scored.index)[i])

        self.ia_kri_mapping = pd.DataFrame({'IA Name': all_ia,
            'KRI Description': all_feats,
            'Quartile value': all_quartiles})
        self.ia_kri_mapping['KRI Description'] = self.ia_kri_mapping['KRI Description'].map(gui_name_dict)

        # append IA ID to mapping table
        self.ia_kri_mapping = pd.merge(self.ia_kri_mapping,self.hit_list_expanded[['IA Name','IA ID']],on='IA Name',how='inner')

        # append kri ID to mapping table
        self.ia_kri_mapping = pd.merge(self.ia_kri_mapping,self.kri_details,on='KRI Description',how='inner')

        # output IA KRI mapping table to excel
        ia_kri_mapping_cols = ['IA ID','KRI ID','Quartile value']
        self.ia_kri_mapping_out = self.ia_kri_mapping[ia_kri_mapping_cols].sort_values(['IA ID','KRI ID'])
        #self.ia_kri_mapping_out.to_excel(f'ia_kri_mapping_{this_run_id}.xlsx',index=False)
        #self.ia_kri_mapping_out.to_excel(self.pickled_dir + '/ia_kri_mapping_02.xlsx',index=False)
        self.ia_kri_mapping_out.to_sql('ia_kri_mapping_02', con = engine, if_exists = 'replace',index = False, chunksize = 1000)
        pickle.dump(self.kri_details, open(self.pickled_dir + '/kri_details.pkl', 'wb'))
        pickle.dump(self.ia_kri_mapping, open(self.pickled_dir + '/ia_kri_mapping.pkl', 'wb'))
        pickle.dump(self.ia_kri_mapping_out, open(self.pickled_dir + '/ia_kri_mapping_out.pkl', 'wb'))
