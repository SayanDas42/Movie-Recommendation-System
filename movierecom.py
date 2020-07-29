import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns



le = LabelEncoder()

url='movies.csv'
dataset=pd.read_csv(url)
df=pd.read_csv(url)

dataset['Country']=dataset['Country'].astype(str)
dataset['Actor 1']=dataset['Actor 1'].astype(str)
dataset['Actor 2']=dataset['Actor 2'].astype(str)

print(dataset.head())
print(dataset.shape)
print('\n')
print('Null Value Count:\n')
print(dataset.isnull().sum())
print('\n')
print(dataset.info())
print('\n')
print(dataset.describe())
print('\n')

dataset['Country']=le.fit_transform(dataset['Country'])
dataset['Language']=le.fit_transform(dataset['Language'])
dataset['Director']=le.fit_transform(dataset['Director'])
dataset['Writer']=le.fit_transform(dataset['Writer'])
dataset['Production Company']=le.fit_transform(dataset['Production Company'])
dataset['Actor 1']=le.fit_transform(dataset['Actor 1'])
dataset['Actor 2']=le.fit_transform(dataset['Actor 2'])
dataset['Genre 1']=le.fit_transform(dataset['Genre 1'])
dataset['Genre 2']=le.fit_transform(dataset['Genre 2'])
dataset['Genre 3']=le.fit_transform(dataset['Genre 3'])

dataset['Watched']=dataset['Watched'].map({'Yes':1,'No':0})


x=dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]]
y=dataset.iloc[:,14]

sns.heatmap(x.corr(), annot=True)
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)


'''print("x_train\n")
print(x_train.head())
print('\n')
print("y_train\n")
print(y_train.head())
print('\n')'''

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


plt.figure(figsize=(20,7))
sns.countplot(x="Genre 1",data=df,palette="Set1")
plt.ylabel("No. of Movies",size=15)
plt.xlabel("Genre",size=15)
plt.title("No. of Movies of each Genre",size=20,pad=20)
plt.show()

plt.figure(figsize=(30,10))
rest = sns.countplot(x="Year",data=df,palette="Set1")
rest.set_xticklabels(rest.get_xticklabels(), rotation=90, ha="right")
plt.ylabel("No. of Movies",size=20)
plt.xlabel("Year",size=20)
plt.title("No. of Movies based on Year",size=20,pad=20)
plt.show()

plt.figure(figsize=(10,10))
#print(df['Rating'].value_counts().sort_values(ascending=True))
ratingtype=df['Rating'].value_counts().sort_values(ascending=True)
slices=[ratingtype[3],ratingtype[4],ratingtype[1],ratingtype[5],ratingtype[2]]
labels=['3 star','4 star','1 star','5 star','2 star']
colors=['green','yellow','grey','red','blue']
plt.pie(slices,colors=colors,labels=labels,autopct='%1.3f%%',pctdistance=0.5,labeldistance=1.2,shadow=False)
plt.title("Percentage of Movies of different ratings",size=10,pad=20)
plt.show()
print('\n')

print('Movies Watched\n')

k=0
c=0
for i in y_train:
	if i==1:
		print(df.index[y_train.index[k]+2],df['Movie Name'][y_train.index[k]],' (Genre :',df['Genre 1'][y_train.index[k]],', Rating :',df['Rating'][y_train.index[k]],')')
		c=c+1
		if c==5:
			break
	k=k+1
print('\n')

print('Movies Not Watched:\n')

k=0
c=0
for i in y_train:
	if i==0:
		print(df.index[y_train.index[k]+2],df['Movie Name'][y_train.index[k]],' (Genre :',df['Genre 1'][y_train.index[k]],', Rating :',df['Rating'][y_train.index[k]],')')
		c=c+1
		if c==5:
			break
	k=k+1
print('\n')


#print(x_test)
#print('\n')
#print(y_test)
#print(y_pred)

print('Recommended Movies:\n')

c=0
for i in range(len(y_pred)):
	if y_pred[i]==1:
		print(df.index[y_test.index[i]+2],df['Movie Name'][y_test.index[i]],' (Genre :',df['Genre 1'][y_test.index[i]],', Rating :',df['Rating'][y_test.index[i]],')')
		c=c+1
		if c==5:
			break
print('\n')


print ("Accuracy using Logistic Regression : ", accuracy_score(y_pred,y_test))
print('\n')

print("Confusion Matrix:\n")
print(confusion_matrix(y_pred,y_test))
print('\n')

print("Classification Report:\n")
print(classification_report(y_pred,y_test))
print('\n')

from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(x_train, y_train)
y_pred = nvclassifier.predict(x_test)
print ("Accuracy using Naive-Bayes : ", accuracy_score(y_pred,y_test))
print('\n')

