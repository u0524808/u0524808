
# coding: utf-8

# In[1]:

from urllib.request import urlopen
from contextlib import closing
url = 'http://aima.cs.berkeley.edu/data/iris.csv'


# In[2]:

with closing(urlopen(url)) as u, open('iris.csv', 'w') as f:
    iris_file = u.read()
    f.write(iris_file.decode('utf-8'))


# In[3]:

from numpy import genfromtxt ,zeros
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))


# In[4]:

print (data)


# In[5]:

target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)

t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3


# In[6]:

print (target)


# In[7]:

#clf=classifier
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(data, t)


# In[8]:

#print(clf.predict(data[0]))
#print(t[0])


# In[9]:

from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(data, t, test_size=0.4, random_state=0)
clf.fit(train,t_train) # train
print(clf.score(test,t_test)) # test
print()


# In[10]:

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(clf.predict(test),t_test))
print()


# In[11]:

from sklearn.metrics import classification_report
print(classification_report(clf.predict(test), t_test, target_names=['setosa', 'versicolor', 'virginica']))
print()


# In[ ]:



