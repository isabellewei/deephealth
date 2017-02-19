from flask import Flask, render_template, request, flash
import pickle
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
app.config["ASSETS_DEBUG"] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True
f1 = open("clf1.pkl", 'rb')
f2 = open("clf2.pkl", 'rb')
f3 = open("maps.pkl")
clf1 = joblib.load(f1)
clf2 = joblib.load(f2)
maps = pickle.load(f3)
map2 = {1:'Emergency',
2:'Urgent',
3:'Elective',
4:'Newborn',
5:'Not Available',
6:'NULL',
7:'Trauma Center',
8:'Not Mapped'}
map3 = {1: 'Discharged to home',
2: 'Discharged/transferred to another short term hospital',
3: 'Discharged/transferred to SNF',
4: 'Discharged/transferred to ICF',
5: 'Discharged/transferred to another type of inpatient care institution',
6: 'Discharged/transferred to home with home health service',
7: 'Left AMA',
8: 'Discharged/transferred to home under care of Home IV provider',
9: 'Admitted as an inpatient to this hospital',
10: 'Neonate discharged to another hospital for neonatal aftercare',
11: 'Expired',
12: 'Still patient or expected to return for outpatient services',
13: 'Hospice / home',
14: 'Hospice / medical facility',
15: 'Discharged/transferred within this institution to Medicare approved swing bed',
16: 'Discharged/transferred/referred another institution for outpatient services',
17: 'Discharged/transferred/referred to this institution for outpatient services',
18: 'NULL',
19: '"Expired at home. Medicaid only, hospice."',
20: '"Expired in a medical facility. Medicaid only, hospice."',
21: '"Expired, place unknown. Medicaid only, hospice."',
22: 'Discharged/transferred to another rehab fac including rehab units of a hospital .',
23: 'Discharged/transferred to a long term care hospital.',
24: 'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
25: 'Not Mapped',
26: 'Unknown/Invalid',
30: 'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
27: 'Discharged/transferred to a federal health care facility.',
28: 'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
29: 'Discharged/transferred to a Critical Access Hospital (CAH).'}
f1.close()
f2.close()
f3.close()
titles = ['encounter_id','patient_nbr','race','gender','age','weight','admission_source_id','time_in_hospital','payer_code','medical_specialty','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum','A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change','diabetesMed','readmitted']


@app.route('/input', methods=['GET', 'POST'])
def main():
    text = ""
    if request.method=='POST':
        inputs = request.form.get("data", "")
        values = [x.strip() for x in inputs.split(',')]
        for i in range(len(titles)):
            if values[i] == '?' or values[i] == None:
                values[i] = -1
            if titles[i] in maps:
                try:
                    values[i] = maps[titles[i]][values[i]]
                except:
                    pass
        pred1 = clf1.predict([values])
        admission = int(pred1[0])
        values.insert(6, admission)
        pred2 = clf2.predict([values])
        discharge = int(pred2[0])
        text = "Admission Type {}: {}\nDischarge Disposition {}: {}".format(admission, map2[int(admission)],
                                                                            discharge, map3[int(discharge)])
    return render_template("input.html", text=text)

app.debug = True
app.secret_key = "testing"

if __name__ == "__main__":
    app.run()
