import pandas as pd
import numpy as np
import pickle
import time
import lime
import lime.lime_tabular
from pathlib import Path
from math import ceil
from sklearn.ensemble import IsolationForest

from traice import batchstep

class Fit(batchstep.BatchStep):

    def __init__(self, input_dir, pickled_dir, out_path,cred_dict):

        super().__init__(input_dir, pickled_dir, out_path,cred_dict)

        # Inputs
        self.df_ia_agg = None
        self.df_ia_agg_std = None

        # Outputs
        self.df_ia_agg_scored = None

    def score_fn(self, df, iforest):
        '''
        Return a score between 0 and 1 with exponential decrease
        COEF can be tuned to result in different score distributions
        '''
        COEF = 10
        dfn = iforest.decision_function(df)

        return 1 / (1 + np.exp(COEF * dfn))

    # Model and Explainer
    def run_step(self):
        import pymysql
        import mysql.connector
        from sqlalchemy import create_engine
        engine = create_engine("mysql+pymysql://" + self.cred_dict['username'] + ":" + self.cred_dict['password'] + "@" + "localhost" + "/" + "traice2")
        self.df_ia_agg = pickle.load(open(self.pickled_dir + '/df_ia_agg.pkl', 'rb'))
        self.df_ia_agg_std = pickle.load(open(self.pickled_dir + '/df_ia_agg_std.pkl', 'rb'))

        # Remove any non - populated columns
        for col in list(self.df_ia_agg_std.isnull().sum()[self.df_ia_agg_std.isnull().sum() > 1].index):
            del self.df_ia_agg_std[col]

        # produce isolation forest model and fit to dataframe
        #iforest = IsolationForest(n_estimators=1000, contamination=0.05)
        print('populated columns',self.df_ia_agg_std.columns)
        iforest = IsolationForest(n_estimators=1000, contamination=0.05, random_state=123)
        iforest.fit(self.df_ia_agg_std)

        # create new dataframe with output scores
        self.df_ia_agg_scored = self.df_ia_agg_std.copy()

        # get raw score from isolation forest model
        self.df_ia_agg_scored['score'] = iforest.decision_function(self.df_ia_agg_std)

       
        # get custom score between 0 and 1 using scoring function
        self.df_ia_agg_scored['risk_score'] = self.score_fn(self.df_ia_agg_std, iforest)

        print('populated columns score',self.df_ia_agg_scored['risk_score'])

        # save isolation forest model
        #with open(f'iforest_{this_run_id}.p', 'wb') as f:
        with open(f'iforest_02.p', 'wb') as f:
            pickle.dump(iforest, f)

        # TODO: sort this out
        """
        from plotnine import *
        % matplotlib
        inline
        (ggplot(self.df_ia_agg_scored, aes('risk_score')) + geom_histogram(bins=100))
​        """

        # define outliers as having risk score > 0.8
        outliers = self.df_ia_agg_scored[self.df_ia_agg_scored['risk_score'] >0.3]
        len(outliers)

        # looking at the IAs in descending order of risk score
        outliers_sorted = outliers.sort_values('risk_score',ascending=False)
        outliers_sorted.head()

        # fit explainer model (not currently used)
        feat_names = list(self.df_ia_agg.columns)
        print("......outliers......",outliers)
        explainer = lime.lime_tabular.LimeTabularExplainer(self.df_ia_agg_std.values,
            feature_names=feat_names,
            class_names=['score'],
            verbose=True,
            mode='regression')
        print("oveerrrrrrrr...................................................")
        # example explainer instance
        #exp = explainer.explain_instance(self.df_ia_agg_scored.loc[outliers.sort_values('score',ascending=False).index[0],:'Clients_With_More_Than_One_KYC_Change'].values,iforest.decision_function,num_features=10)

        # example explainer features with negative contributions to anomaly score
        #[r for r in exp.as_list() if r[1] < 0]

        # example explainer features in notebook format
        #exp.show_in_notebook(show_table=True)

        # get explainer results for each instance
        # DBG Load from pickle to avoid this step, which takes about 1.5 hours to run on a laptop. 
        self.df_ia_agg_scored['explainer_results'] = self.df_ia_agg_std.apply(lambda x: explainer.explain_instance(x.values,iforest.decision_function,num_features=10),axis=1)
        pickle.dump(self.df_ia_agg_scored, open(self.pickled_dir + '/df_ia_agg_scored_explainer.pkl', 'wb'))
        self.df_ia_agg_scored = pickle.load(open(self.pickled_dir + '/df_ia_agg_scored_explainer.pkl', 'rb'))

        # list of explainer results (not currently used)
        self.df_ia_agg_scored['exp_list'] = self.df_ia_agg_scored['explainer_results'].apply(lambda x: x.as_list())

        # list of KRIs sorted by number of standard deviations from mean
        self.df_ia_agg_scored['largest_list'] = self.df_ia_agg_scored.iloc[:,:-4].apply(lambda x: list(self.df_ia_agg_scored.columns[x.values.argsort()[-5:][::-1]]),axis=1)

        ## Dumping pre-generated dataframes

        #import pickle

        print('df_ia_agg_scored is',self.df_ia_agg_scored)
        #self.df_ia_agg_scored.to_excel(self.pickled_dir + '/df_ida_agg_scored.xlsx', index=False)


        self.df_ia_agg_scored.iloc[:,:24].to_sql('df_ia_agg_scored', con = engine, if_exists = 'replace',index = False, chunksize = 1000)

        #with open(f'df_ia_agg_{this_run_id}.p','wb') as f:
        with open(f'df_ia_agg_02.p','wb') as f:
            pickle.dump(self.df_ia_agg_scored,f)

        pickle.dump(self.df_ia_agg_scored, open(self.pickled_dir + '/df_ia_agg_scored.pkl', 'wb'))
