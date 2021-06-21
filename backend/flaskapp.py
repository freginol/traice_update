
from flask import Flask, jsonify, request
import pickle
import pandas as pd
import os
app = Flask(__name__)

PICKLED_DIR = '../csvfiles/pickled/batch'
branch_binned = pickle.load(open(PICKLED_DIR + '/branch_binned.pkl', 'rb'))
hit_list = pickle.load(open(PICKLED_DIR + '/hit_list.pkl', 'rb'))
hit_list_out = pickle.load(open(PICKLED_DIR + '/hit_list_out.pkl', 'rb'))
hit_list_expanded_out = pickle.load(open(PICKLED_DIR + '/hit_list_expanded_out.pkl', 'rb'))
ia_kri_mapping_out = pickle.load(open(PICKLED_DIR + '/ia_kri_mapping_out.pkl', 'rb'))
df_ia_agg_scored = pickle.load(open(PICKLED_DIR + '/df_ia_agg_scored.2.pkl', 'rb'))
wellbeing1 = pickle.load(open(PICKLED_DIR + '/wellbeing.pkl', 'rb'))

print(df_ia_agg_scored)
print(type(df_ia_agg_scored))
print('dsdfsdfsdfsdfsdfsfd',df_ia_agg_scored.to_dict('list'))
@app.route('/hitlist')
def hitlist():

    return jsonify(hit_list.to_dict('list')), 800

@app.route('/branchbinned')
def branchbinned():
    return jsonify(branch_binned.to_dict('list')), 800


@app.route('/hitlistout')
def hitlistout():
    return jsonify(hit_list_out.to_dict('list')), 800

@app.route('/hitlistexpandedout')
def hitlistexpandedout():
    return jsonify(hit_list_expanded_out.to_dict('list')), 800

@app.route('/iakrimapping')
def iakrimapping():
    return jsonify(ia_kri_mapping_out.to_dict('list')), 800

@app.route('/aggscored')
def aggscored():
    return str(df_ia_agg_scored.to_dict('list'))


@app.route('/wellbeing')
def wellbeing():
    return jsonify(wellbeing1.to_dict('list')), 800

@app.route('/api/predict', methods=['POST'])
def predict():
    print('one')
    login_json = request.get_json()

    if not login_json:
        return jsonify({'msg': 'Missing JSON'}), 400

    step = login_json.get('step')
    type1 = login_json.get('type')
    amount=login_json.get('amount')
    newbalanceDest= login_json.get('newbalanceDest')
    oldbalanceDest=login_json.get('oldbalanceDest')
    newbalanceOrig=login_json.get('newbalanceOrig')
    oldbalanceOrg=login_json.get('oldbalanceOrg')

    if not step:
        return jsonify({'prediction': 'step is missing'}), 400

    if not type1:
        return jsonify({'prediction': 'type is missing'}), 400


    if not amount:
        return jsonify({'prediction': 'amount is missing'}), 400

    if not newbalanceDest:
        return jsonify({'prediction': 'newbalancedest is missing'}), 400


    if not oldbalanceDest:
        return jsonify({'prediction': 'oldbalanceDest is missing'}), 400

    if not newbalanceOrig:
        return jsonify({'prediction': 'newbalanceOrig is missing'}), 400
    
    if not oldbalanceOrg:
        return jsonify({'prediction': 'oldbalanceOrg is missing'}), 400
    

    
    x_unit_test=pd.DataFrame([[int(step),type1,float(amount),float(oldbalanceOrg),float(newbalanceOrig),float(oldbalanceDest),float(newbalanceDest)]],columns=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
       'oldbalanceDest', 'newbalanceDest'])
    unique_types=['CASH_IN', 'PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']

    for each_categorical_value in unique_types:
        #x_unit_test[each_categorical_value]=0
        #print(each_categorical_value)
        if(x_unit_test['type'][0]==each_categorical_value):
            x_unit_test[each_categorical_value]=1
        else:
            x_unit_test[each_categorical_value]=0
        #x_unit_test[each_categorical_value]=pd.get_dummies
    x_unit_test=x_unit_test[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
       'oldbalanceDest', 'newbalanceDest','CASH_IN','PAYMENT', 'TRANSFER', 'CASH_OUT',
       'DEBIT']]
    x_unit_test.drop(columns=['type'],inplace=True)
    print('prediction1',x_unit_test.head())
    sc=pickle.load(open('../../traice_moneylaundering/scaler.pkl','rb'))
    x_unit_test_scales=sc.transform(x_unit_test)
    rf=pickle.load(open('../../traice_moneylaundering/GradientBoostingClassifier()_best_model.pkl','rb'))
    print('prediction3',x_unit_test.head())
    prediction=rf.predict(x_unit_test_scales)
    print('prediction4',prediction)
    return jsonify({'prediction': str(prediction[0])}), 800

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='0.0.0.0')

