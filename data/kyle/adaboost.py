# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:30:04 2017

@author: kylem_000
"""
def adaboost(min_samples_split = [2], min_samples_leaf = [1], n_estimators = [50], num_trials = 1,
             training_size = 0.01, test_size = 0.1, max_depth = None):
    
    from time import time
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score
    import tables
    import numpy as np
    import scipy as sp
    import pandas as pd
    
    filename = 'adaboost.npy'
    data = np.load(filename)
    data_headings = np.array(['acc1', 'acc2', 'min_samples_split', 'min_samples_leaf', 'n_estimators', 'trials', 'training_size', 'test_size'])
    
    df = pd.read_csv("parsed.csv")
    y1 = df["admission_type_id"].values
    y2 = df["discharge_disposition_id"].values
    columns = list(df)[1:4] + list(df)[7:49]
    X = df[columns].values
    
    for m_s_s in min_samples_split:
        for m_s_l in min_samples_leaf:
            for n_est in n_estimators:          
        
                acc1 = np.empty([1,num_trials])
                acc2 = np.empty([1,num_trials])
                
                t0 = time()
                
                for i in range(0, num_trials):
                    X1_big, X1_train, y1_big, y1_train = train_test_split(X, y1, test_size = training_size)
                    X2_big, X2_train, y2_big, y2_train = train_test_split(X, y2, test_size = training_size)
                    X1_waste, X1_test, y1_waste, y1_test = train_test_split(X1_big, y1_big, test_size = (test_size)/(1.01-training_size))
                    X2_waste, X2_test, y2_waste, y2_test = train_test_split(X2_big, y2_big, test_size = (test_size)/(1.01-training_size))
                    
                    DT = DecisionTreeClassifier(max_depth = max_depth, min_samples_split = m_s_s, min_samples_leaf = m_s_l)
                    
                    clf1 = AdaBoostClassifier(base_estimator = DT, n_estimators = n_est)
                    clf2 = AdaBoostClassifier(base_estimator = DT, n_estimators = n_est)
                    clf1.fit(X1_train, y1_train)
                    clf2.fit(X2_train, y2_train)
                    y1_pred = clf1.predict(X1_test)
                    y2_pred = clf2.predict(X2_test)
                    acc1[0,i] = accuracy_score(y1_test, y1_pred)
                    acc2[0,i] = accuracy_score(y2_test, y2_pred)
                    print i
                print "accuracy1 mean:", np.mean(acc1), "std:", np.std(acc1)
                print "accuracy2 mean:", np.mean(acc2), "std:", np.std(acc2)
                print "runtime:", round(time()-t0,3)
                print "DT min_samples_split:", m_s_s
                print "DT min_samples_leaf:", m_s_l
                print "AB n_estimators:", n_est
                new_data = np.array([[np.mean(acc1), np.mean(acc2), m_s_s, m_s_l, n_est, num_trials, training_size, test_size]])
                data = np.concatenate((data, new_data), 0)
                np.save(filename, data)

def get_data():
    import numpy as np
    filename = 'adaboost.npy'
    data = np.load(filename)
    return data

def find_max():
    import numpy as np
    data = get_data()
    print "max1:", data[np.argmax(data[:,0]),:]
    print "max2:", data[np.argmax(data[:,1]),:]
    

