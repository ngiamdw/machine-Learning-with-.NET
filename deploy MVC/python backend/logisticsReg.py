import pickle
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# import matplotlib.pyplot as plt
# import seaborn as sb

#read data file
df = pd.read_csv('iris-data-clean.csv')

#rename
df = df.rename(columns = {"sepal_length_cm":"SepalLength","sepal_width_cm":"SepalWidth","petal_length_cm":"PetalLength","petal_width_cm":"PetalWidth","class":"Class"})

boolEncode = {
    "Setosa":0,
    "Versicolor":1,
    "Virginica":2
}
df["Class"] = df["Class"].map(boolEncode)

# Training fours features (Sepal Length,width & petal length,width) 
X = df.loc[:,"SepalLength":"PetalWidth"]

print("max array\n",np.max(X,axis = 0))
print("min arr\n",np.min(X,axis = 0))
# label
y = df['Class']
print(X,"\n",y,"\n")
print(X.loc[130])
#Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# need to specify multi_class = 'multinomial' and solver
logReg = LogisticRegression(solver = 'lbfgs',random_state = 42,max_iter= 3000)
logReg.fit(X_train.values, y_train)  

#Model validation
y_pred = logReg.predict(X_test)
print("unique(y_pred)",np.unique(y_pred))
print("test input",logReg.predict([[4.9,3.0,1.5,1.2]]))
print(accuracy_score(y_test, y_pred))
DefaultAccuracy = accuracy_score(y_test, y_pred) #accuracy based on dataset

# deploy to the flask server flask server need to be started
pickle.dump(DefaultAccuracy, open('LRAcc.pkl', 'wb'))  #serialize the object
pickle.dump(logReg, open('logReg.pkl', 'wb'))  


#original_df.to_pickle("./dummy.pkl")
#unpickled_df = pd.read_pickle("./dummy.pkl")

#Sample input to get "0"
#?SepalLength=4.9&SepalWidth=3.0&PetalLength=1.5&PetalWidth=1.2
#Sample input to get "2"
#SepalLength = 7.7, SepalWidth = 3.0, PetalLength = 6.1 , PetalWidth = 2.3