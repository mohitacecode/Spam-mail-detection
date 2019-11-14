import pandas as pd
import numpy as np
df = pd.read_csv('../DataSet/smsspamcollection.tsv',sep='\t')
df.head()
df.isnull().sum()
df.dropna(inplace=True)
blanks = []
for i,lb,rv,l,p in df.itertuples():
    if(type(rv) == str):
        if(rv.isspace()):
            blanks.append(i);
df.drop(blanks, inplace = True)

X = df['message']
y = df['label']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

Text_Class_SVC = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC()),])

Text_Class_SVC.fit(X_train, y_train)

pred = Text_Class_SVC.predict(X_test)

from sklearn import metrics
print("Accuracy Score = " + str(metrics.accuracy_score(y_test,pred))+"\n")
print("Confusion Matrix = " + str(metrics.confusion_matrix(y_test,pred))+"\n")
print("Classification Report = " + str(metrics.classification_report(y_test,pred))+"\n")

