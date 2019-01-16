#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:15:10 2019

@author: kiraliang
"""
import pickle
import pandas as pd
import json
import sys

MODEL_PATH = "./LR_model.sav"
THRESHOLD = 50

def load_model():
    model = pickle.load(open(MODEL_PATH, 'rb'))
    
    return model

def predict(model,df):
    y_pred = model.predict_proba(df)
    #dis_score = round(y_pred[0][0]*100,2)
    sat_score = round(y_pred[0][1]*100,2)
    result = {'sat_score':sat_score}
    print(json.dumps(result))
#    if dis_score > THRESHOLD:
#        print('{"dissatisfied":"' + str(dis_score) + '"}')
#    else:
#        print('{"satisfied":"' + str(sat_score) + '"}')
def main():
    file_name = sys.argv[1]
    model = load_model()
    features_df = pd.read_csv(file_name + ".csv").iloc[:,1:]
    predict(model,features_df)
    
if __name__ == "__main__":
    main()