
import pandas as pd

df = pd.read_csv('diabetes-training.csv')
del df['encounter_id']
nan = df['weight'][0]
cols = list(df)
numerical = set(['patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
                 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses'])
text_cols = [x for x in cols if x not in numerical]
for cat in text_cols:
    options = df[cat].unique()
    mapping = dict(zip(options, range(len(options))))
    mapping[nan] = -1
    df = df.replace({cat: mapping})

df = df.fillna(-1)
print df.isnull().values.any()
df.to_csv('parsed.csv',index=False)