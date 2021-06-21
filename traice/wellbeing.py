import pickle
import numpy as np
import pandas as pd
from pathlib import Path
#from plotnine import *
from textblob import TextBlob
from traice import batchstep

class Wellbeing(batchstep.BatchStep):

    def __init__(self, input_dir, pickled_dir, out_path,cred_dict):

        super().__init__(input_dir, pickled_dir, out_path,cred_dict)

        # Inputs
        self.hit_list = None

        # Outputs
        self.wellbeing_df = None

    def run_step(self):
            
        # TODO check this
        np.random.seed(123)

        self.hit_list = pickle.load(open(self.pickled_dir + '/hit_list.pkl', 'rb'))
        self.emails = pickle.load(open(self.pickled_dir + '/emails.pkl', 'rb'))
        # create template columns for IA behaviour
        ia_repeated = np.repeat(self.hit_list['IA ID'].values,12)
        months_repeated = np.tile([5,5,5,6,6,6,7,7,7,8,8,8],len(self.hit_list['IA ID'].values))
        desc_repeated = np.tile(['anxiety','fulfillment','judgement'],int(len(ia_repeated)/3))

        def sentiment(x):
            if(x>0):
                return 'fulfillment'
            elif(x<0):
                return 'anxiety'
            else:
                return 'judgement'
        
    
        #(ggplot(data=df_test) + geom_line(aes('i','a')) + geom_line(aes('i','f')) + geom_line(aes('i','j')) + scale_x_continuous(limits=[8,16]))

        self.emails=self.emails[['text','IA ID','Month']]

        self.emails['Value']=self.emails['text'].apply(lambda x: TextBlob(str(x)).polarity)

        self.emails['Metric description']=self.emails['Value'].apply(sentiment)
        # output wellbeing values to excel
        self.wellbeing = pd.DataFrame({'IA ID':self.emails['IA ID'],
            'Month': self.emails['Month'],
            'Metric description': self.emails['Metric description'],
            'Value':self.emails['Value']})
        #self.wellbeing.to_excel(f'wellbeing_{this_run_id}.xlsx',index=False)
        
        import pymysql
        import mysql.connector
        from sqlalchemy import create_engine
        engine = create_engine("mysql+pymysql://" + self.cred_dict['username'] + ":" + self.cred_dict['password'] + "@" + "localhost" + "/" + "traice2")
        self.wellbeing.to_sql('wellbeing', con = engine, if_exists = 'replace',index = False, chunksize = 1000)
        #self.wellbeing.to_excel(self.pickled_dir + '/wellbeing_02.xlsx',index=False)

        pickle.dump(self.wellbeing, open(self.pickled_dir + '/wellbeing.pkl', 'wb'))
