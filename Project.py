#!/usr/bin/env python
# coding: utf-8

# In[1]:


# please first install these below libraries then proceed.  
# pip install plotly
# pip install cufflinks


# <h1>Libraries</h1>

# In[2]:


import numpy as np
import pandas as pd
import pandas
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from colorama import Fore, Back, Style 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import plotly.graph_objs as go
from plotly.offline import plot, iplot, init_notebook_mode
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as gobj
import plotly.figure_factory as ff
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, make_scorer
warnings.filterwarnings("ignore")


# <h1>Data Analysis</h1>

# In[3]:


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print("Data:\n")
print(df.head())


# In[4]:


df.isnull().values.any()


# In[5]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), vmin=-1, annot=True);


# In[6]:


k = df.loc[:,["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","high_blood_pressure","platelets",
                  "serum_creatinine","serum_sodium","sex","smoking","time"]]
k["index"] = np.arange(1,len(k)+1)
scat_plot = ff.create_scatterplotmatrix(k, diag='box', index='index',colormap='Portland', colormap_type='cat',
                                        height=2400, width=1800)
iplot(scat_plot)


# In[7]:


fig = px.histogram(df, x="platelets", color="DEATH_EVENT")
fig.show()


# In[8]:


trace1 = go.Bar(
                x = df.DEATH_EVENT, y = df.diabetes, name = "Diabetes",
                marker = dict(color = 'rgba(255, 182, 50, 0.5)',), text = df.diabetes)

trace2 = go.Bar(
                x = df.DEATH_EVENT, y = df.high_blood_pressure, name = "High Blood Pressure",
                marker = dict(color = 'rgba(255, 178, 200, 0.5)',), text = df.high_blood_pressure)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[9]:


dff = df[["age", "anaemia", "creatinine_phosphokinase", "ejection_fraction", "high_blood_pressure",
          "platelets", "serum_creatinine", "serum_sodium", "time"]]
print(dff.head())


# In[10]:


# Total number of female survived
f_s = df[(df["sex"]==0) & (df["DEATH_EVENT"]==0)]["DEATH_EVENT"].shape[0]
# Total number of female not survived
f_n_s = df[(df["sex"]==0) & (df["DEATH_EVENT"]==1)]["DEATH_EVENT"].shape[0]
# Total number of male survived
m_s = df[(df["sex"]==1) & (df["DEATH_EVENT"]==0)]["DEATH_EVENT"].shape[0]
# Total number of male not survived
m_n_s = df[(df["sex"]==1) & (df["DEATH_EVENT"]==1)]["DEATH_EVENT"].shape[0]
labels = ['Male - Survived','Male - Not Survived', "Female -  Survived", "Female - Not Survived"]
values = [m_s, m_n_s, f_s, f_n_s]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(title_text="Correlation of Heart Failure with predictor-Gender")
fig.show()


# In[11]:


asd = pd.DataFrame(columns=["Age Group", "Survived", "Not Survived"])
age_group = []
m = []
n = []
for i in range(int(df['age'].min()), int(df['age'].max()), 5):
    if i < int(df['age'].max()):
        age_group.append(str(i)+'-'+str(i+5))
        m.append(df[(df["age"]>=i) & (df["age"]<i+5)]["DEATH_EVENT"].value_counts().values[0])
        n.append(df[(df["age"]>=i) & (df["age"]<i+5)]["DEATH_EVENT"].value_counts().values[1])
    else:
        break
asd['Age Group'] = age_group
asd['Survived'] = m
asd['Not Survived'] = n
ax = asd.plot(x='Age Group', marker='o', title = "Survival by Age Group")
ax.set_xlabel("Age Group")
ax.set_ylabel("Count")


# In[12]:


trace1 = go.Bar(
                x = df.DEATH_EVENT, y = df.smoking, name = "Smoking",
                marker = dict(color = 'rgba(255, 182, 50, 0.5)',), text = df.smoking)

trace2 = go.Bar(
                x = df.DEATH_EVENT, y = df.high_blood_pressure, name = "High Blood Pressure",
                marker = dict(color = 'rgba(128, 300, 200, 0.5)',), text = df.high_blood_pressure)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[13]:


trace1 = go.Bar(
                x = df.DEATH_EVENT, y = df.anaemia, name = "Anaemia",
                marker = dict(color = 'rgba(255, 182, 50, 0.5)',), text = df.anaemia)

trace2 = go.Bar(
                x = df.DEATH_EVENT, y = df.sex, name = "Sex",
                marker = dict(color = 'rgba(255, 178, 200, 0.5)',), text = df.sex)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[14]:


jd = df.drop('DEATH_EVENT', axis=1)
print(jd.head())


# In[15]:

print("Description of Data:\n")
print(df.describe())


# <h1>Feature Selection</h1>

# In[16]:


from sklearn.feature_selection import SelectKBest


# In[17]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=12)
embeded_rf_selector.fit(jd, df["DEATH_EVENT"])

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = jd.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


# In[18]:


embeded_rf_feature


# In[19]:

print("Dataset after feature selection:")
print(df[embeded_rf_feature].head())


# In[20]:


Features = ['age', 'ejection_fraction','serum_creatinine' , 'time']
x = df[Features]
y = df["DEATH_EVENT"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2, stratify=y)


# In[21]:


y_test.value_counts()


# In[22]:


y_train.value_counts()


# <h1>Data Modeling</h1>

# In[23]:


x_train.shape, y_train.shape


# In[24]:


x_test.shape, y_test.shape


# <h3>K-Nearest Neighbors</h3>

# In[25]:


# KNN
# Hyper-tunning the 'n_neighbors' parameter
neighbors = list(range(1,21,1))
knn_accuracy_test = []
knn_accuracy_train = []
knn_df = pd.DataFrame(columns=['n_neighbors', 'test_Accuracy', 'train_accuracy'])
for i in neighbors:
    knn_clf = KNeighborsClassifier(n_neighbors=i)
    knn_clf.fit(x_train, y_train)
    knn_pred = knn_clf.predict(x_test)
    knn_acc = accuracy_score(y_test, knn_pred)
    knn_accuracy_test.append(knn_acc)
    knn_pred = knn_clf.predict(x_train)
    knn_acc = accuracy_score(y_train, knn_pred)
    knn_accuracy_train.append(knn_acc)
    
knn_df['n_neighbors'] = neighbors
knn_df['test_Accuracy'] = knn_accuracy_test
knn_df['train_accuracy'] = knn_accuracy_train
knn_df['difference'] = abs(knn_df['test_Accuracy'].values - knn_df['train_accuracy'].values)
k_best, k_accu_best = knn_df.sort_values(['difference', 'test_Accuracy'], ascending=[True, False]).head(1)[['n_neighbors', 'test_Accuracy']].values[0]
print("Best Parameter for KNN, k =", k_best)
plt.plot(neighbors, knn_accuracy_test, label='test data', marker='o')
plt.plot(neighbors, knn_accuracy_train, label='train data', marker='o')
plt.plot(k_best, k_accu_best, marker='o', color='red')
plt.grid()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.title("For K-Nearest Neighbors")
plt.legend(loc='upper right')
plt.show()


# <h3>Decision Tree</h3>

# In[26]:


# Decision Tree
# Hyper-tunning the 'criterion', 'max_depth', 'min_sample_leaf' and 'min_sample_split' parameter
criterion = ['gini', 'entropy']
max_depth = [1,2,3,4,5]
min_sample_leaf = [1,2,3,4,5]
min_sample_split = [2,3,4,5]
dt_accuracy_test = []
dt_accuracy_train = []
parameter = []
dt_df = pd.DataFrame(columns=['Parameters', 'test_Accuracy', 'train_Accuracy'])
count = 0
cnt = []
for i in criterion:
    for j in max_depth:
        for k in min_sample_leaf:
            for l in min_sample_split:
                count+=1
                dt_clf = DecisionTreeClassifier(criterion=i, max_depth=j, min_samples_leaf=k, min_samples_split=l)
                dt_clf.fit(x_train, y_train)
                dt_pred = dt_clf.predict(x_test)
                dt_acc = accuracy_score(y_test, dt_pred)
                dt_accuracy_test.append(dt_acc)
                dt_pred = dt_clf.predict(x_train)
                dt_acc = accuracy_score(y_train, dt_pred)
                dt_accuracy_train.append(dt_acc)
                cnt.append(count)
                para = '[criterion = '+str(i)+', max_depth = '+str(j)+', min_sample_leaf = '+str(k)+', min_sample_split = '+str(l)+']'
                parameter.append(para)

dt_df['Parameters'] = parameter
dt_df['test_Accuracy'] = dt_accuracy_test
dt_df['train_Accuracy'] = dt_accuracy_train
dt_df['difference'] = abs(dt_df['test_Accuracy'] - dt_df['train_Accuracy'])
dt_df["count"] = cnt

dt_best, dt_accu_best, best_para = dt_df.sort_values(['difference', 'test_Accuracy'], ascending=[True, False]).head(1)[['count', 'test_Accuracy', 'Parameters']].values[0]
print("Best Parameter for Decision Tree =", best_para)
plt.plot(cnt, dt_accuracy_test, label='test_data')
plt.plot(cnt, dt_accuracy_train, label='train_data')
plt.plot(dt_best, dt_accu_best, marker='o', color='red')
plt.grid()
plt.xlabel("Various Parameters")
plt.ylabel("Accuracy")
plt.title("For Decision-Tree")
plt.legend(loc='upper left')
plt.show()


# <h3>Random Forest</h3>

# In[27]:


# Random Forest
rf_accuracy_test = []
rf_accuracy_train = []
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)
rf_pred = rf_clf.predict(x_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Best Accuracy for Random Forest:", rf_acc)


# <h3>Support Vector Machine</h3>

# In[28]:


# SVM(SVC)
# Hyper-tunning the 'kernel' and 'gamma' parameter
svc_dict = {'auto': ['linear'], 'scale': ['linear', 'poly', 'rbf', 'sigmoid']}
svc_accuracy_test = []
svc_accuracy_train = []
svc_df = pd.DataFrame(columns=['Parameters', 'test_Accuracy', 'train_Accuracy'])
parameter = []
cnt = []
count = 0
for i in svc_dict:
    for j in svc_dict[i]:
        count+=1
        svc_clf = SVC(kernel=j, gamma=i)
        svc_clf.fit(x_train, y_train)
        svc_pred = svc_clf.predict(x_test)
        svc_acc = accuracy_score(y_test, svc_pred)
        svc_accuracy_test.append(svc_acc)
        svc_pred = svc_clf.predict(x_train)
        svc_acc = accuracy_score(y_train, svc_pred)
        svc_accuracy_train.append(svc_acc)
        cnt.append(count)
        para = '[kernel = '+str(i)+', gamma = '+str(j)+']'
        parameter.append(para)
svc_df['Parameters'] = parameter
svc_df['test_Accuracy'] = svc_accuracy_test
svc_df['train_Accuracy'] = svc_accuracy_train
svc_df['difference'] = abs(svc_df['test_Accuracy'] - svc_df['train_Accuracy'])
svc_df["count"] = cnt

svc_best, svc_accu_best, best_para = svc_df.sort_values(['difference', 'test_Accuracy'], ascending=[True, False]).head(1)[['count', 'test_Accuracy', 'Parameters']].values[0]
print('Best Parameter for Support Vector Machine(SVC) = ', best_para)
plt.plot(cnt, svc_accuracy_test, label='test_data', marker ='o')
plt.plot(cnt, svc_accuracy_train, label='train_data', marker='o')
plt.plot(svc_best, svc_accu_best, marker='o', color='red')
plt.grid()
plt.xlabel("Various Parameters")
plt.ylabel("Accuracy")
plt.title("For SVC")
plt.legend(loc='upper left')
plt.show()


# <h3>Logistic Regression</h3>

# In[29]:


# Logistic Regression
# Hyper-tunning the 'penalty' and 'solver' parameter
lr_dict = {'l1': ['liblinear', 'saga'], 'l2': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
lr_accuracy_test = []
lr_accuracy_train = []
lr_accuracy_cv = []
lr_df = pd.DataFrame(columns=['Parameters', 'test_Accuracy', 'train_Accuracy'])
parameter = []
cnt = []
count = 0
for i in lr_dict:
    for j in lr_dict[i]:
        count+=1
        lr_clf = LogisticRegression(penalty=i, solver=j)
        lr_clf.fit(x_train, y_train)
        lr_pred = lr_clf.predict(x_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        lr_accuracy_test.append(lr_acc)
        lr_pred = lr_clf.predict(x_train)
        lr_acc = accuracy_score(y_train, lr_pred)
        lr_accuracy_train.append(lr_acc)
        cnt.append(count)
        para = '[penalty = '+str(i)+', solver = '+str(j)+']'
        parameter.append(para)

lr_df['Parameters'] = parameter
lr_df['test_Accuracy'] = lr_accuracy_test
lr_df['train_Accuracy'] = lr_accuracy_train
lr_df['difference'] = abs(lr_df['test_Accuracy'] - lr_df['train_Accuracy'])
lr_df["count"] = cnt

lr_best, lr_accu_best, best_para = lr_df.sort_values(['difference', 'test_Accuracy'], ascending=[True, False]).head(1)[['count', 'test_Accuracy', 'Parameters']].values[0]
print('Best Parameters for Logistic Regression =', best_para)
plt.plot(cnt, lr_accuracy_test, label='test_data', marker ='o')
plt.plot(cnt, lr_accuracy_train, label='train_data', marker='o')
plt.plot(lr_best, lr_accu_best, marker='o', color='red')
plt.grid()
plt.xlabel("Various Parameters")
plt.ylabel("Accuracy")
plt.title("For Logistic Regression")
plt.legend(loc='upper left')
plt.show()


# <h3>Model Comparison</h3>

# In[30]:


all_accuracy = [dt_accu_best, k_accu_best, lr_accu_best, rf_acc, svc_accu_best]
models = ["Decision-Tree", "KNN", "Logistic", "Random-Forest", "SVM"]
data = {'Models':models, 'Accuracy':all_accuracy}
dfg = pd.DataFrame(data, columns=['Models',"Accuracy"])
print(dfg)
#display(dfg.style.apply(lambda x: ['background: lightblue' if i == max(dfg["Accuracy"]) else '' for i in dfg["Accuracy"]]))


# In[31]:


axx = dfg.plot(x="Models", marker="o")
axx.set_ylabel('Accuracy')
plt.grid()
plt.show()


# <h1>Predictions</h1>

# In[32]:


dt_df.sort_values(['difference', 'test_Accuracy'], ascending=[True, False]).head(1)


# In[33]:


best_clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=2, min_samples_split=2)
best_clf.fit(x_train, y_train)
best_pred = best_clf.predict(x_test)
x_test["Predicted_DEATH_EVENT"] = best_pred
print("Predictions:")
print(x_test.head())


# <h1>Confusion Matrix</h1>

# In[34]:


cf_matrix = confusion_matrix(y_test, best_pred)
group_names = ['True Pos', 'False Neg', 'False Pos', 'True Neg']
group_counts = cf_matrix.flatten()
group_percentages = np.round(cf_matrix.flatten()/sum(cf_matrix.flatten()), 2)
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[ ]:




