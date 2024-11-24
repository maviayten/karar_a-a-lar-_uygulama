#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


iris = load_iris()
X, y = iris.data, iris.target  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[2]:


dtree=DecisionTreeClassifier(random_state=42)
dtree.fit(X_train,y_train)


# In[3]:


y_pred = dtree.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Karmaşıklık Matrisi:\n", conf_matrix)

class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("Sınıflandırma Raporu:\n", class_report)


precision = precision_score(y_test, y_pred, average='macro')
print("Hassasiyet:", precision)

recall = recall_score(y_test, y_pred, average='macro')
print("Duyarlılık:", recall)


f1 = f1_score(y_test, y_pred, average='macro')
print("F1 Skoru:", f1)


# In[4]:


plt.figure(figsize=(30,20))
plot_tree(dtree,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.show()


# In[5]:


dtree_tuned = DecisionTreeClassifier(
    criterion='entropy',
    splitter='random',
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=3,
    max_features='log2',
    random_state=42,
    max_leaf_nodes=20,
    min_impurity_decrease=0.01
)
dtree_tuned.fit(X_train, y_train)


# In[6]:


y_pred2 = dtree_tuned.predict(X_test)

accuracy = accuracy_score(y_test, y_pred2)
print("Doğruluk:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred2)
print("Karmaşıklık Matrisi:\n", conf_matrix)

class_report = classification_report(y_test, y_pred2, target_names=iris.target_names)
print("Sınıflandırma Raporu:\n", class_report)

precision = precision_score(y_test, y_pred2, average='macro')
print("Hassasiyet:", precision)

print("Duyarlılık:", recall)


f1 = f1_score(y_test, y_pred2, average='macro')
print("F1 Skoru:", f1)


# In[9]:


plt.figure(figsize=(90,50))
plot_tree(dtree_tuned, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()


# In[ ]:




