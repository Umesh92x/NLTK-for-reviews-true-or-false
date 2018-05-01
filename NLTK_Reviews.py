# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    
for i in range(0,1000):
    # Importing the dataset
    review=re.sub(r'[^a-zA-z]',' ',dataset['Review'][i])
    review=review.lower()
    review =review.split() 
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy

# apply Rednom Forest
from sklearn.ensemble import RandomForestClassifier

classifier1=RandomForestClassifier()
classifier1.fit(x_train,y_train)
y_pred=classifier1.predict(x_test)
cm_forest=confusion_matrix(y_test,y_pred)

# Apply KNN
from sklearn.neighbors import KNeighborsClassifier
classifier2=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier2.fit(x_train,y_train)
y_pred_KNN=classifier.predict(x_test)

cm_KNN=confusion_matrix(y_test,y_pred_KNN)









