# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:22:08 2024

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from tkinter import ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import VotingClassifier
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from statistics import stdev
import matplotlib.pyplot as plt





dataset = arff.loadarff('dataset_31_credit-g.arff')
df = pd.DataFrame(dataset[0])
df.head()

print(df)

for col in df:
    if pd.api.types.is_object_dtype(df[col]):
        try:
            df[col] = df[col].astype(str)
        except ValueError:
            pass

#print("Redovi sa nedostajucom vrednoscu: \n")
#print(df[df.isna().any(axis=1)])

renamed_columns = {
    'checking_status': 'Existing checking account',
    'duration': 'Duration [months]',
    'credit_history': 'Credit history',
    'purpose': 'Purpose',
    'credit_amount': 'Credit amount',
    'savings_status': 'Status of savings account',
    'employment': 'Employment',
    'installment_commitment': 'Installment rate',
    'personal_status': 'Personal status and sex',
    'other_parties': 'Other debtors / guarantors',
    'residence_since': 'Present residence',
    'property_magnitude': 'Property',
    'age': 'Age',
    'other_payment_plans': 'Other installment plans',
    'housing': 'Housing',
    'existing_credits': 'Existing credits',
    'job': 'Job',
    'num_dependents': 'Number of people being liable to provide maintenance for',
    'own_telephone': 'Telephone',
    'foreign_worker': 'Foreign worker',
    'class': 'Good or bad'
}


df = df.rename(columns=renamed_columns)
#print(df.columns)
print("Tipovi podataka", df.dtypes)

good_bad = df['Good or bad'].unique()
print(good_bad)

df['Good or bad'] = df['Good or bad'].map({'good': 1.0, 'bad': 0.0})

X = df.drop(columns=("Good or bad"), axis=1)
Y = df["Good or bad"]

print('Employment categories: ', df["Employment"].unique())
employee_categories = df["Employment"].value_counts().reset_index()

print('Existing credits: ', df["Existing credits"].unique())
existing_credits_categories = df["Existing credits"].value_counts().reset_index()

print('Credit history: ', df["Credit history"].unique())
credit_history_categories = df["Credit history"].value_counts().reset_index()

print('Age categories: ', df["Age"].unique())
age_categories = df["Age"].value_counts().reset_index()

print('Property categories: ', df["Property"].unique())
property_categories = df["Property"].value_counts().reset_index()

print('Job categories: ', df["Job"].unique())
property_categories = df["Job"].value_counts().reset_index()

# Filtriranje podataka za odobrene i neodobrene kredite
approved = df[df['Good or bad'] == 1]
not_approved = df[df['Good or bad'] == 0]

# Brojanje po kategorijama za obe podskupine
approved_counts = approved['Employment'].value_counts()
not_approved_counts = not_approved['Employment'].value_counts()



# Kreiranje stubičastog dijagrama za odobrene kredite po radnom stazu
length_of_service = plt.figure(figsize=(12, 6))
plt.bar(approved_counts.index, approved_counts.values, color='b')
plt.xlabel('Length of service')
plt.ylabel('Number of approved credits')
plt.title('Number of approved credits by length of service')
plt.xticks(rotation=45)
plt.show()

# Brojanje po kategorijama za obe podskupine
approved_counts = approved['Existing credits'].value_counts()
not_approved_counts = not_approved['Existing credits'].value_counts()

total_credit_history_counts = df['Credit history'].value_counts()
approved_credit_history_counts = approved['Credit history'].value_counts()

# Osiguravanje da sve kategorije budu u oba skupa podataka
approved_credit_history_counts = approved_credit_history_counts.reindex(total_credit_history_counts.index, fill_value=0)

# Priprema podataka za grafikon
bar_width = 0.4
index = range(len(total_credit_history_counts.index))

# Kreiranje stubičastog dijagrama
credit_history = plt.figure(figsize=(12, 6))
plt.bar(index, total_credit_history_counts.values, bar_width, label='Total number of credits', color='gray', align='center')
plt.bar([i + bar_width for i in index], approved_credit_history_counts.values, bar_width, label='Number of approved credits', color='pink', align='center')

plt.xlabel('Credit history')
plt.ylabel('Number of credits')
plt.title('Total and approved credits by credit history category')
plt.xticks([i + bar_width / 2 for i in index], total_credit_history_counts.index, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

'''# Kreiranje stubičastog dijagrama za odobrene kredite po radnom stazu
plt.figure(figsize=(12, 6))
plt.bar(approved_counts.index, approved_counts.values, color='r')
plt.xlabel('Number of existing credits')
plt.ylabel('Number of approved credits')
plt.title('Number of approved credits by number of existing credits')
plt.xticks(rotation=45)
plt.show()

existing_credits = df['Existing credits'].value_counts()

# Kreiranje stubičastog dijagrama za odobrene kredite po radnom stazu
plt.figure(figsize=(12, 6))
plt.bar(existing_credits.index, existing_credits.values, color='y')
plt.xlabel('Number of existing credits')
plt.ylabel('Number of approved credits')
plt.title('Number of existing credits')
plt.xticks(rotation=45)
plt.show()'''

# Brojanje po nekretninama
total_property_counts = df['Property'].value_counts()
approved_property_counts = approved['Property'].value_counts()

# Osiguravanje da sve kategorije budu u oba skupa podataka
approved_property_counts = approved_property_counts.reindex(total_property_counts.index, fill_value=0)

# Priprema podataka za grafikon
bar_width = 0.4
index = range(len(total_property_counts.index))

# Kreiranje stubičastog dijagrama
type_of_property = plt.figure(figsize=(12, 6))
plt.bar(index, total_property_counts.values, bar_width, label='Total number of credits', color='yellow', align='center')
plt.bar([i + bar_width for i in index], approved_property_counts.values, bar_width, label='Number of approved credits', color='purple', align='center')

plt.xlabel('Type of property')
plt.ylabel('Number of credits')
plt.title('Total and аpproved credits by property category')
plt.xticks([i + bar_width / 2 for i in index], total_property_counts.index, rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# Brojanje po kategorijama 'Existing credits'
approved_counts = approved['Existing credits'].value_counts().sort_index()
not_approved_counts = not_approved['Existing credits'].value_counts().sort_index()
existing_credits_counts = df['Existing credits'].value_counts().sort_index()

# Kreiranje linijskog dijagrama
existing_credits = plt.figure(figsize=(12, 6))
plt.plot(existing_credits_counts.index, existing_credits_counts.values, marker='o', color='y', label='Total number of people')
plt.plot(approved_counts.index, approved_counts.values, marker='o', color='purple', label='Number of approved credits')

plt.xlabel('Number of existing credits')
plt.ylabel('Count')
plt.title('Comparison of total number of people and approved credits by number of existing credits')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Kreiranje tortnih grafikona za udeo odobrenih i neodobrenih kredita po kategorijama 'Existing credits'
categories = existing_credits_counts.index

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.flatten()

for i, category in enumerate(categories[:4]):  # Prikazujemo samo prva 4 tipa kredita radi bolje vidljivosti
    total_count = existing_credits_counts[category]
    approved_count = approved_counts.get(category, 0)
    not_approved_count = not_approved_counts.get(category, 0)

    sizes = [approved_count, not_approved_count]
    labels = ['Approved', 'Not approved']
    colors = ['lightgreen', 'lightcoral']

    axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Existing credits: {category}')
    axes[i].axis('equal')  # Podrazumevano čini da je krug krug, ne elipsa

plt.suptitle('Distribution of approved and not approved credits by number of existing credits', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

df['Age category'] = pd.cut(df['Age'], bins=[0, 24, 65, float('inf')], labels=['<24', '24-65', '65+'])
print(df[['Age', 'Age category']].head())

approved_cases_age = df[df['Good or bad'] == 1]
not_approved_cases_age = df[df['Good or bad'] == 0]

approved_age_counts = approved_cases_age['Age category'].value_counts().sort_index()
not_approved_age_counts = approved_cases_age['Age category'].value_counts().sort_index()


# Kreiranje stubičastog dijagrama za odobrene kredite po godinama
credits_by_age = plt.figure(figsize=(12, 6))
plt.bar(approved_age_counts.index, approved_age_counts.values, color='g')
plt.xlabel('Age category')
plt.ylabel('Number of approved credits')
plt.title('Number of approved credits by age category')
plt.show()


# Grupisanje podataka prema kategorijama zaposlenja i starosnim kategorijama
grouped_data = df.groupby(['Employment', 'Age category']).size().unstack(fill_value=0)

# Kreiranje grafikona za svaku kategoriju zaposlenja
fig, ax = plt.subplots(figsize=(12, 8))
grouped_data.plot(kind='bar', stacked=True, ax=ax)

plt.title('Number of people who were approved for a loan by age category and length of service')
plt.xlabel('Length of service')
plt.ylabel('Number of people')
plt.legend(title='Age category')
plt.xticks(rotation=45)
plt.show()

categorical_features = ["Existing checking account", "Credit history", 
                         "Purpose","Status of savings account",
                         "Employment", "Personal status and sex",
                         "Other debtors / guarantors", "Property", 
                         "Other installment plans", "Housing", "Job",
                         "Telephone", "Foreign worker"]

one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                    one_hot,
                                    categorical_features)],
                                    remainder = "passthrough")
transformed_X = transformer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(transformed_X, Y, 
                                                    test_size=0.20, 
                                                    random_state=40)
#%% Linearna regresija
regr = LinearRegression()
regr.fit(X_train, Y_train)


print("Linear Regression Score:", regr.score(X_test, Y_test))

Y_pred = regr.predict(X_test)


linear_graph = plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.plot([0, 1], [0, 1], 'k--', lw=2)  
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.title('Real vs predicted values')
plt.show()
#%%Logisticka regresija
log_reg = LogisticRegression(solver='newton-cholesky',
                             penalty='l2',
                             max_iter=500,
                             random_state=20)
log_reg.fit(X_train, Y_train)

Y_pred = log_reg.predict(X_test)


print("Logistic regression score:", log_reg.score(X_test, Y_test))
print("Logistic regression classification report:\n", classification_report(Y_test, Y_pred))

Y_pred = log_reg.predict(X_test)
Y_prob = log_reg.predict_proba(X_test)[:, 1]

# Računanje ROC krive
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
roc_auc = roc_auc_score(Y_test, Y_prob)

# Plotovanje ROC krive
logistic_graph = plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Logistic regression (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# Matrica konfuzije
cm = confusion_matrix(Y_test, Y_pred)
disp_logistic = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_)
disp_logistic.plot(cmap=plt.cm.Blues)
plt.title('Confusion matrix for logistic regression')
plt.show()

# Именовање категорија након трансформације
feature_names = transformer.get_feature_names_out()

# Коефицијенти логистичке регресије
coefficients = log_reg.coef_[0]

# Креирање DataFrame-а за преглед карактеристика и њихових коефицијената
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Сортирање по апсолутној вредности коефицијената
importance_df['Absolute Coefficient'] = np.abs(importance_df['Coefficient'])
importance_df = importance_df.sort_values(by='Absolute Coefficient', ascending=False)

# Приказ првих неколико најважнијих карактеристика
print(importance_df.head(10))
#%% Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
Y_prob = knn.predict_proba(X_test)[:, 1]

print("KNN accuracy:", accuracy_score(Y_test, Y_pred))
print("KNN classification report:\n", classification_report(Y_test, Y_pred))

# ROC kriva
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
roc_auc = roc_auc_score(Y_test, Y_prob)


knn_graph = plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('KNN (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# Matrica konfuzije
cm_knn = confusion_matrix(Y_test, Y_pred)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn.classes_)
disp_knn.plot(cmap=plt.cm.Blues)
plt.title('Confusion matrix for knn')
plt.show()
#%% XGBoost
xgb_model = xgb.XGBClassifier(
    max_depth=8,  # Dubina stabala
    learning_rate=0.5,  # Stopa učenja
    n_estimators=150,  # Broj stabala
    min_child_weight=1,  # Minimalna težina deteta (za kontrolu prenaučenosti)
    gamma=0,  # Parametar za kontrolu prenaučenosti
    subsample=0.8,  # Udeo podataka za učenje za svaki model
    colsample_bytree=0.8,  # Udeo atributa za svako stablo
    objective='binary:logistic',  # Ciljna funkcija za binarnu klasifikaciju
    random_state=42)
xgb_model.fit(X_train, Y_train)
Y_pred = xgb_model.predict(X_test)
print("XGBoost accuracy:", accuracy_score(Y_test, Y_pred))
print("XGBoost classification report:\n", classification_report(Y_test, Y_pred))

Y_prob = xgb_model.predict_proba(X_test)[:, 1]

# Računanje ROC krive
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
roc_auc = roc_auc_score(Y_test, Y_prob)

# Prikazivanje ROC krive
roc_xgb = plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('XGBoost (ROC) curve')
plt.legend(loc="lower right")
plt.show()
#%% voting classifier soft 
ensemble_model_soft = VotingClassifier(estimators=[('lr', log_reg), ('xgb', xgb_model), ('knn', knn)], voting='soft')

ensemble_model_soft.fit(X_train, Y_train)

y_pred = ensemble_model_soft.predict(X_test)

print("Ensemble Model Accuracy:", accuracy_score(Y_test, y_pred))
print("Ensemble Model Classification Report:\n", classification_report(Y_test, y_pred))

score2 = cross_val_score(ensemble_model_soft, X_train, Y_train, cv=5, scoring='recall')

VC_soft_cv_score = score2.mean()
VC_soft_cv_stdev = stdev(score2)

print('Cross Validation Recall scores are: {}'.format(score2))
print('Average Cross Validation Recall score: ', VC_soft_cv_score)
print('Cross Validation Recall standard deviation: ', VC_soft_cv_stdev)

cm = confusion_matrix(Y_test, y_pred)
disp_soft = ConfusionMatrixDisplay(confusion_matrix=cm)
disp_soft.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Ensemble Model (Soft Voting)')
plt.show()


print("Ensemble Model Classification Report:\n", classification_report(Y_test, y_pred))

# Bar plot for cross-validation recall scores
x = np.arange(5)
cross_val = plt.bar(x, score2)
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Recall Score')
plt.title('Cross-Validation Recall Scores for Ensemble Model (Soft Voting)')
plt.axhline(y=VC_soft_cv_score, color='r', linestyle='--', label='Average Score')
plt.legend()
plt.show()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    ensemble_model_soft, X_train, Y_train, cv=5, scoring='recall',
    train_sizes=np.linspace(0.1, 1.0, 10))

learning_curve_graph = plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Recall Score')
plt.title('Learning Curve for Ensemble Model (Soft Voting)')
plt.legend()
plt.show()

#%%% voting classifier hard

ensemble_mode_hardl = VotingClassifier(estimators=[('lr', log_reg), ('xgb', xgb_model), ('knn', knn)], voting='hard')

ensemble_mode_hardl.fit(X_train, Y_train)

y_pred = ensemble_mode_hardl.predict(X_test)

print("Ensemble Model Accuracy:", accuracy_score(Y_test, y_pred))
print("Ensemble Model Classification Report:\n", classification_report(Y_test, y_pred))


score = cross_val_score(ensemble_mode_hardl, X_train, Y_train, cv=5, scoring='recall')
VC_hard_cv_score = score.mean()
VC_hard_cv_stdev = stdev(score)
print('Cross Validation Recall scores are: {}'.format(score))
print('Average Cross Validation Recall score: ', VC_hard_cv_score)
print('Cross Validation Recall standard deviation: ', VC_hard_cv_stdev)


cm = confusion_matrix(Y_test, y_pred)
disp_hard = ConfusionMatrixDisplay(confusion_matrix=cm)
disp_hard.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Ensemble Model (Soft Voting)')
plt.show()

#%% 
def configure_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))


def zoom(event):
    if event.delta > 0:  # Zoom in
        canvas.scale("all", 0, 0, 1.1, 1.1)
    else:  # Zoom out
        canvas.scale("all", 0, 0, 0.9, 0.9)
    configure_scroll_region(None)
    
    
root = tk.Tk()
root.title("Assessment of candidate's creditworthiness")
root.state('zoomed')

canvas = tk.Canvas(root)
canvas.pack(side="left", fill="both", expand=True)

vsb = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
vsb.pack(side="right", fill="y")
canvas.configure(yscrollcommand=vsb.set)

hsb = ttk.Scrollbar(root, orient="horizontal", command=canvas.xview)
hsb.pack(side="bottom", fill="x")
canvas.configure(xscrollcommand=hsb.set)

main_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=main_frame, anchor="nw")

main_frame.bind("<Configure>", configure_scroll_region)

frames = []
header_frame = tk.Frame(main_frame, bd=2, relief="groove", bg="lightblue")
header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

header_label = tk.Label(header_frame, text="Credit analisys dashboard", font=("Helvetica", 24, "bold"), bg="lightblue")
header_label.pack(pady=10)
frames.append(header_frame)

feame_width = 200
frame_height = 300

first_frame = tk.Frame(main_frame, bd=2, relief="groove", background="white",width=feame_width, height=frame_height)
first_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
frames.append(first_frame)

second_frame = tk.Frame(main_frame, bd=2, relief="groove", background="white",width=feame_width, height=frame_height)
second_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
frames.append(second_frame)

third_frame = tk.Frame(main_frame, bd=2, relief="groove", background="white",width=feame_width, height=frame_height)
third_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
frames.append(third_frame)

forth_frame = tk.Frame(main_frame, bd=2, relief="groove", background="white",width=feame_width, height=frame_height)
forth_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
frames.append(forth_frame)

fifth_frame = tk.Frame(main_frame, bd=2, relief="groove", background="white",width=feame_width, height=frame_height)
fifth_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ew")
frames.append(fifth_frame)

sixth = tk.Frame(main_frame, bd=2, relief="groove", background="white",width=feame_width, height=frame_height)
sixth.grid(row=6, column=0, padx=10, pady=10, sticky="ew")
frames.append(sixth)

seveth = tk.Frame(main_frame, bd=2, relief="groove", background="white",width=feame_width, height=frame_height)
seveth.grid(row=7, column=0, padx=10, pady=10, sticky="ew")
frames.append(seveth)

eight = tk.Frame(main_frame, bd=2, relief="groove", background="white",width=feame_width, height=frame_height)
eight.grid(row=8, column=0, padx=10, pady=10, sticky="ew")
frames.append(eight)

canvas1 = FigureCanvasTkAgg(length_of_service, master = first_frame)
canvas1.draw()
canvas1.get_tk_widget().pack(side="left",  expand=True, pady=(10, 5))

canvas2 = FigureCanvasTkAgg(credit_history, master = first_frame)
canvas2.draw()
canvas2.get_tk_widget().pack(side="left", fill="y", expand=True, pady=(10, 5))

canvas3 = FigureCanvasTkAgg(type_of_property, master = second_frame)
canvas3.draw()
canvas3.get_tk_widget().pack(side="bottom", fill="y", expand=True, pady=(10, 5))

canvas4 = FigureCanvasTkAgg(existing_credits, master = third_frame)
canvas4.draw()
canvas4.get_tk_widget().pack(side="left", fill="y", expand=True, pady=(10, 5))

canvas5 = FigureCanvasTkAgg(credits_by_age, master = third_frame)
canvas5.draw()
canvas5.get_tk_widget().pack(side="bottom", fill="y", expand=True, pady=(10, 5))

canvas7 = FigureCanvasTkAgg(linear_graph, master = forth_frame)
canvas7.draw()
canvas7.get_tk_widget().pack(side="left", fill="y", expand=True, pady=(10, 5))

canvas8 = FigureCanvasTkAgg(logistic_graph, master = fifth_frame)
canvas8.draw()
canvas8.get_tk_widget().pack(side="left", fill="y", expand=True, pady=(10, 5))

canvas9 = FigureCanvasTkAgg(disp_logistic.figure_, master = fifth_frame)
canvas9.draw()
canvas9.get_tk_widget().pack(side="left", fill="y", expand=True, pady=(10, 5))

canvas10 = FigureCanvasTkAgg(knn_graph, master = sixth)
canvas10.draw()
canvas10.get_tk_widget().pack(side="left", fill="y", expand=True, pady=(10, 5))

canvas11= FigureCanvasTkAgg(disp_knn.figure_, master = sixth)
canvas11.draw()
canvas11.get_tk_widget().pack(side="left", fill="y", expand=True, pady=(10, 5))

canvas12 = FigureCanvasTkAgg(roc_xgb, master = sixth)
canvas12.draw()
canvas12.get_tk_widget().pack(side="bottom", fill="y", expand=True, pady=(10, 5))

canvas13 = FigureCanvasTkAgg(disp_soft.figure_, master = seveth)
canvas13.draw()
canvas13.get_tk_widget().pack(side="left", fill="y", expand=True, pady=(10, 5))

canvas14 = FigureCanvasTkAgg(disp_hard.figure_, master = seveth)
canvas14.draw()
canvas14.get_tk_widget().pack(side="left", fill="y", expand=True, pady=(10, 5))

# canvas15 = FigureCanvasTkAgg(cross_val, master = fifth_frame)
# canvas15.draw()
# canvas15.get_tk_widget().pack(side="bottom", fill="both", expand=True, pady=(10, 5))

canvas16 = FigureCanvasTkAgg(learning_curve_graph, master = eight)
canvas16.draw()
canvas16.get_tk_widget().pack(side="bottom", fill="y", expand=True, pady=(10, 5))


canvas.bind("<MouseWheel>", zoom)

configure_scroll_region(None)
root.mainloop()