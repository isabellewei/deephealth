from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

df = pd.read_csv("../parsed.csv")
y1 = df["admission_type_id"].values
y2 = df["discharge_disposition_id"].values
columns = list(df)[1:4] + list(df)[7:49]
X = df[columns].values

#ignore this(just takinig out some columns)
cols = list(df)
dropped = set(['admission_type_id', 'discharge_disposition_id', 'weight', 'payer_code'])
columns2 = [z for z in cols if z not in dropped]
X2 = df[columns2].values

X_train, X_test, y_train, y_test = train_test_split(X,y1,test_size=0.3)


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(50,25), warm_start=True,
                    batch_size=20, solver='sgd', activation='relu', alpha=0.001,
                    learning_rate='adaptive', learning_rate_init=0.01,
                    max_iter=1000
                    )
#train 1
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(classification_report(y_test, predictions))

#train 2
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(classification_report(y_test, predictions))







