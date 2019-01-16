#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:09:18 2019

@author: kiraliang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:10:57 2019

@author: kiraliang
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd

FEATURES_PATH = "./features.csv"
FILENAME = './LR_model.sav'

def read_features():
    '''
    Read features csv file from the file_path
    '''
    features = pd.read_csv(FEATURES_PATH)
    
    return features

def data_splitting(df):
    '''
    Split data in train:75% and test:25%
    '''
    X = df.iloc[:,1:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,shuffle=True)
    
    return X_train, X_test, y_train, y_test

def LogisticRegression_train(X_train, X_test, y_train, y_test, save_model=True):
    classifier = LogisticRegression(random_state=0,solver="liblinear")
    classifier.fit(X_train, y_train)
    #y_pred = classifier.predict(X_test)
    acc = classifier.score(X_test, y_test)
    print("Accuracy of logistic regression classifier: {:.2f}".format(acc))
    if save_model:
        pickle.dump(classifier, open(FILENAME, 'wb'))
        
        
def main():
    features = read_features()
    X_train, X_test, y_train, y_test = data_splitting(features)
    LogisticRegression_train(X_train, X_test, y_train, y_test)
    print("Training of model is done!")
     
if __name__ == "__main__":
    main()
