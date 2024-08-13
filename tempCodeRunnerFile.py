import pandas as pd
df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
x=df.drop(['Churn'],axis=1)
y=df['Churn']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)
import joblib
joblib.dump(model,'model_saved')