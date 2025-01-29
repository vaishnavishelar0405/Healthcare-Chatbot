import streamlit as st
import mysql.connector
import pandas as pd
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pyttsx3
import re
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Establish connection to MySQL database
conn = mysql.connector.connect(
    host='127.0.0.1',
    user="root",
    password="SQL@123",
    database="healthcare_database"
)

# Create cursor object
cursor = conn.cursor()

# Fetch necessary data from the database
cursor.execute("SELECT id, name, age, gender, symptoms FROM patients")
patients_data = cursor.fetchall()

# Close cursor and connection
cursor.close()
conn.close()

# Convert fetched data to DataFrame
columns = ['id', 'name', 'age', 'gender', 'symptoms']
patients_df = pd.DataFrame(patients_data, columns=columns)

training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

scores = cross_val_score(clf, x_test, y_test, cv=3)

model = SVC()
model.fit(x_train, y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp) + 1) > 13):
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."

def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

def getInfo():
    name = st.text_input("Your Name")
    age = st.text_input("Your Age")
    gender = st.selectbox("Your Gender", ["Male", "Female", "Other"])
    symptoms_input = st.text_input("Symptoms ")

    if st.button("Submit"):
        symptoms = [symptom.strip() for symptom in symptoms_input.split(",")]
        insert_query = "INSERT INTO patients (name, age, gender, symptoms) VALUES (%s, %s, %s, %s)"
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user="root",
            password="SQL@123",
            database="healthcare_database"
        )
        cursor = conn.cursor()
        cursor.execute(insert_query, (name, age, gender, ','.join(symptoms)))
        conn.commit()
        cursor.close()
        conn.close()
        return symptoms

def check_pattern(dis_list, inp):
    pred_list = []
    inp = ','.join(inp)  # Convert list of symptoms to a comma-separated string
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return True, pred_list
    else:
        return False, []

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def tree_to_code(tree, feature_names, disease_input):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    #while True:
       # disease_input = st.text_input("Enter the symptom you are experiencing")
        #conf_inp = disease_input
        #conf, cnf_dis = check_pattern(chk_dis, disease_input)
        #if conf:
            #for num, it in enumerate(cnf_dis):
              #  st.write(f"{num}: {it}")
           ## if len(cnf_dis) > 0:
               # conf_inp = st.number_input(f"Select the one you meant (0 - {len(cnf_dis)-1})", min_value=0, max_value=len(cnf_dis)-1)
                #disease_input = cnf_dis[int(conf_inp)]
           # break
       # else:
           # st.write("Enter valid symptom.")

    while True:
        try:
            num_days = st.number_input("From how many days?", min_value=1, step=1)
            break
        except:
            st.write("Enter valid input.")

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = st.radio(f"Are you experiencing {syms}?", options=["Yes", "No"])
                if(inp == "Yes"):
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            result = calc_condition(symptoms_exp, num_days)
            if(present_disease[0] == second_prediction[0]):
                st.write("You may have ", present_disease[0])
                st.write(description_list[present_disease[0]])

            else:
                st.write("You may have ", present_disease[0], "or ", second_prediction[0])
                st.write(description_list[present_disease[0]])
                st.write(description_list[second_prediction[0]])

            precution_list = precautionDictionary[present_disease[0]]
            st.write("Take following measures : ")
            for i, j in enumerate(precution_list):
                st.write(i + 1, ")", j)

    recurse(0, 1)

getSeverityDict()
getDescription()
getprecautionDict()
symptoms = getInfo()
tree_to_code(clf, cols, symptoms)
