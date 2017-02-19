# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:30:04 2017

@author: kylem_000
"""
def random_forest(n_estimators = [10], crit = [1], max_features = [35],
                  max_depth = [None], min_samples_split = [2], min_samples_leaf = [1],
                              n_trials = 1, train_size = 0.02, test_size = 0.2):
    
    from time import time
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import numpy as np
    import scipy as sp
    import pandas as pd
    
    filename = 'random_forest.npy'
    data = np.load(filename)
    data_headings = np.array(['acc1', 'acc2', 'n_estimators', 'crit', 'max_features',
                              'max_depth', 'min_samples_split', 'min_samples_leaf', 'n_trials', 'train_size', 'test_size'])
    
    df = pd.read_csv("parsed.csv")
    y1 = df["admission_type_id"].values
    y2 = df["discharge_disposition_id"].values
#    columns = list(df)[0:4] + list(df)[8:9] +list(df)[12:49]
    columns = list(df)[0:5] + list(df)[8:49]
    X = df[columns].values
          
    n_sweep = len(n_estimators)*len(crit)*len(max_features)*len(max_depth)*len(min_samples_split)*len(min_samples_leaf)
    n_counter = 0
    
    for n_est in n_estimators:
        for c in crit:
            for m_f in max_features:
                for m_d in max_depth:
                    for m_s_s in min_samples_split:
                        for m_s_l in min_samples_leaf:             
  
                            if c == 0:
                                criterion = "gini"
                            if c == 1:
                                criterion = "entropy"
                                
                                
                            acc1 = np.empty([1,n_trials])
                            acc2 = np.empty([1,n_trials])
                            
                            t0 = time()
                            
                            for i in range(0, n_trials):
                                
                                X1_train, X1_rest, y1_train, y1_rest = train_test_split(X, y1, train_size = train_size)
                                X2_train, X2_rest, y2_train, y2_rest = train_test_split(X, y2, train_size = train_size)
                                X1_waste, X1_test, y1_waste, y1_test = train_test_split(X1_rest, y1_rest, test_size = test_size/(1.001-train_size))
                                X2_waste, X2_test, y2_waste, y2_test = train_test_split(X2_rest, y2_rest, test_size = test_size/(1.001-train_size))
                                
                                clf1 = RandomForestClassifier(n_estimators = n_est,
                                                              criterion = criterion,
                                                              max_features = m_f,
                                                              max_depth = m_d,
                                                              min_samples_split = m_s_s,
                                                              min_samples_leaf = m_s_l
                                                              )
                                clf2 = RandomForestClassifier(n_estimators = n_est,
                                                              criterion = criterion,
                                                              max_features = m_f,
                                                              max_depth = m_d,
                                                              min_samples_split = m_s_s,
                                                              min_samples_leaf = m_s_l
                                                              ) 
                                clf1.fit(X1_train, y1_train)
                                clf2.fit(X2_train, y2_train)
                                y1_pred = clf1.predict(X1_test)
                                y2_pred = clf2.predict(X2_test)
                                acc1[0,i] = accuracy_score(y1_test, y1_pred)
                                acc2[0,i] = accuracy_score(y2_test, y2_pred)
                                print i
                            n_counter += 1
                            print "accuracy1 mean:", np.mean(acc1), "std:", np.std(acc1)
                            print "accuracy2 mean:", np.mean(acc2), "std:", np.std(acc2)
                            print "runtime:", round(time()-t0,3)
                            print "completion:", n_counter, "/", n_sweep
                            new_data = np.array([[np.mean(acc1), np.mean(acc2), n_est, c, m_f, m_d, m_s_s,
                                                  m_s_l, n_trials, train_size, test_size]])
                            data = np.concatenate((data, new_data), 0)
                            np.save(filename, data)

def init():
    import numpy as np
    filename = 'random_forest.npy'
    data = np.array([[0,0,0,0,0,0,0,0,0,0,0]])
    np.save(filename, data)
    

def get_data():
    import numpy as np
    filename = 'random_forest.npy'
    data = np.load(filename)
    return data

def find_max():
    import numpy as np
    data = get_data()
    np.set_printoptions(threshold=np.nan)
    print "max1:", data[np.argmax(data[:,0]),:]
    print "max2:", data[np.argmax(data[:,1]),:]
    return data[np.argmax(data[:,0]),:], data[np.argmax(data[:,1]),:]

    

