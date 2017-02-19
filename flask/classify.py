import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('../parsed.csv')
labels1 = df['admission_type_id'].values
labels2 = df['discharge_disposition_id'].values
cols = list(df)
f2 = [x for x in cols if x not in set(['discharge_disposition_id'])]
f1 = [x for x in cols if x not in set(['admission_type_id', 'discharge_disposition_id'])]

features1 = df[list(f1)].values
features2 = df[list(f2)].values
f1_train, f1_test, f2_train, f2_test, l1_train, l1_test, l2_train, l2_test = \
    train_test_split(features1, features2, labels1, labels2, test_size=0.25)

# clf = KNeighborsClassifier(10, weights='distance')
# clf1 = AdaBoostClassifier(n_estimators=50)
# clf2 = AdaBoostClassifier(n_estimators=50)
clf1 = RandomForestClassifier(n_jobs=-1, n_estimators=50, min_samples_split=70, max_features='auto')
clf2 = RandomForestClassifier(n_jobs=-1, n_estimators=50, min_samples_split=70, max_features='auto')
# clf1 = GaussianNB()
# clf2 = GaussianNB()
clf1.fit(f1_train, l1_train)
clf2.fit(f2_train, l2_train)

pred1 = clf1.predict(f1_test)
acc1 = accuracy_score(pred1, l1_test)
f1_test = np.insert(f1_test,6, pred1,1)

pred2 = clf2.predict(f1_test)
acc2 = accuracy_score(pred2, l2_test)
print acc1, acc2

from sklearn.externals import joblib
joblib.dump(clf1, 'clf1.pkl')
joblib.dump(clf2, 'clf2.pkl')



