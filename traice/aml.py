import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from traice import batchstep
import pymysql
import mysql.connector
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

class Aml(batchstep.BatchStep):

    def __init__(self, input_dir, pickled_dir, out_path,cred_dict):

        super().__init__(input_dir, pickled_dir, out_path,cred_dict)
    
    def trainFitTest(self,model,space):
        ##models = [lr,knn,svc,nb,dtc,rfc,bc,abc,gbc]
        score = []
        ##for model in models:
        #model.fit(X_train,Y_train)
        cv_inner = KFold(n_splits=10, shuffle=True, random_state=42)
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        #print(type(search))
        result = search.fit(self.X_train, self.y_train)#training the model
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        pickle.dump(best_model, open('./'+str(model)+'_best_model.pkl', 'wb'))
        print('training score',best_model.score(self.X_train,self.y_train))
        print('prediction score - ',best_model.score(self.X_test,self.y_test))
        score.append(best_model.score(self.X_test,self.y_test))
        Y_rfc_pred = best_model.predict(self.X_test)
        print(Y_rfc_pred)
        print(classification_report(self.y_test,Y_rfc_pred))
        print(confusion_matrix(self.y_test,Y_rfc_pred))

    ## Branch Binned
    def run_step(self):

        engine = create_engine("mysql+pymysql://" + self.cred_dict['username'] + ":" + self.cred_dict['password'] + "@" + "localhost" + "/" + "traice2")
        
        self.dataset = pickle.load(open(self.pickled_dir + '/aml.pkl', 'rb'))
        print(self.dataset)

        self.dataset.drop('nameOrig', axis=1, inplace=True)
        self.dataset.drop('nameDest', axis=1, inplace=True)
        self.dataset.drop('isFlaggedFraud', axis=1, inplace=True)

        sample_dataframe = self.dataset.sample(random_state=42,n=100000)
        X=sample_dataframe
        unique_types=X['type'].unique()

        for each_categorical_value in X['type'].unique():
            X[each_categorical_value] = pd.get_dummies(X['type'])[each_categorical_value]

        X.drop('type',axis=True,inplace=True)
        X.head()

        X = sample_dataframe.iloc[:, [0,1,2,3,4,5,7,8,9,10,11]].values
        y = sample_dataframe.iloc[:, 6].values

        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)


        #counts = np.unique(y_train, return_counts=True)

        # Feature Scaling
        
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        pickle.dump(sc, open('./'+'scaler.pkl', 'wb'))
        #X_val = sc.transform(X_val)
        self.X_test = sc.transform(self.X_test)
        #print(counts)

        lr = LogisticRegression()
        knn = KNeighborsClassifier()
        svc = SVC()
        nb = GaussianNB()
        dtc = DecisionTreeClassifier()
        rfc = RandomForestClassifier()
        bc = BaggingClassifier()
        abc = AdaBoostClassifier()
        gbc = GradientBoostingClassifier()

        space=dict()
        space['max_leaf_nodes'] = [10, 25, 500]
        space['max_features'] = [2, 4, 6]
        space['n_estimators']=[10,20,100]
        self.trainFitTest(gbc,space)

        

        
        #space = dict()
        #pace['n_jobs']=[1,2,5]
        #trainFitTest(lr,space)

        #space=dict()
        #space['max_leaf_nodes'] = [10, 25, 500]
        #space['max_features'] = [2, 4, 6]
        #trainFitTest(dtc,space)

        #space=dict()
        #space['max_leaf_nodes'] = [10, 25, 500]
        #space['max_features'] = [2, 4, 6]
        #   trainFitTest(rfc,space)





        #self.aml.to_sql('aml', con = engine, if_exists = 'replace',index = False, chunksize = 1000)
        #pickle.dump(self.branch_binned, open(self.pickled_dir + '/branch_binned.pkl', 'wb'))
        #pickle.dump(self.joined_tables, open(self.pickled_dir + '/joined_tables.pkl', 'wb'))