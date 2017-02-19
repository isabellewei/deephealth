import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('../parsed.csv')
df2 = pd.read_csv('../parsed_problem.csv')
cols_test = list(df2)
f_test = [x for x in cols_test if x not in set(['admission_type_id', 'discharge_disposition_id'])]
f_test = df2[list(f_test)].values

labels1 = df['admission_type_id'].values
labels2 = df['discharge_disposition_id'].values
cols = list(df)
f1 = [x for x in cols if x not in set(['admission_type_id', 'discharge_disposition_id'])]
f2 = [x for x in cols if x not in set(['discharge_disposition_id'])]
features1 = df[list(f1)].values
features2 = df[list(f2)].values

#clf = KNeighborsClassifier(10, weights='distance')
#clf1 = AdaBoostClassifier(n_estimators=50)
#clf2 = AdaBoostClassifier(n_estimators=50)
clf1 = RandomForestClassifier(n_jobs=-1, n_estimators=200, min_samples_split=12, max_features=None)
clf2 = RandomForestClassifier(n_jobs=-1, n_estimators=200, min_samples_split=12, max_features=None)
#clf1 = GaussianNB()
#clf2 = GaussianNB()
clf1.fit(features1, labels1)
clf2.fit(features2, labels2)

pred1 = clf1.predict(f_test)
f_test = np.insert(f_test,6, pred1,1)
pred2 = clf2.predict(f_test)
print df['encounter_id'].values.shape, pred1.shape, pred2.shape
result = np.column_stack((df2['encounter_id'].values, pred1, pred2))
np.savetxt("output.csv", result.astype(int), delimiter=",")
