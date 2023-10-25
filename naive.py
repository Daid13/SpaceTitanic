from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
#PassengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name

def bad_split(inp):
    temp=str(inp).split('/')
    if len(temp)==3:
        return temp
    else:
        print(temp)
        return [None,None,None]

def categoricalise(dataframe):
    
    #print(dataframe)
    dataframe["Cabin1"]=dataframe.apply(lambda row: str(row.Cabin).split('/')[0],axis=1)
    #print(dataframe)
    #dataframe["Cabin2"]=dataframe.apply(lambda row: str(row.Cabin).split('/')[1],axis=1)
    dataframe["Cabin3"]=dataframe.apply(lambda row: str(row.Cabin).split('/')[2],axis=1)
    dataframe["Age1"]=dataframe.apply(lambda row: row.Age//10,axis=1)
    dataframe["RoomService1"]=dataframe.apply(lambda row: row.RoomService>0,axis=1)
    dataframe["FoodCourt1"]=dataframe.apply(lambda row: row.FoodCourt>0,axis=1)
    dataframe["ShoppingMall1"]=dataframe.apply(lambda row: row.ShoppingMall>0,axis=1)
    dataframe["Spa1"]=dataframe.apply(lambda row: row.Spa>0,axis=1)
    dataframe["VRDeck1"]=dataframe.apply(lambda row: row.VRDeck>0,axis=1)
    dataframe.drop(columns=["PassengerId","Cabin", "Age", "RoomService","FoodCourt","ShoppingMall","Spa","VRDeck","Name", "Transported"])
    enc=OrdinalEncoder()
    enc.fit(dataframe)
    temp=pd.DataFrame(enc.transform(dataframe))
    #print(temp)
    return temp


inital=pd.read_csv('data/train.csv')
inital.dropna(inplace=True)
y=pd.DataFrame(inital, columns=["Transported"])
#print(y)

X = categoricalise(inital)
print(X)
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print("hi",X_test)
cnb=CategoricalNB()
y_pred=cnb.fit(X_train,y_train).predict(X_test)
y_PRED=pd.DataFrame(y_pred)
print(y_PRED)
print(y_test)




#print('\nPrediction accuracy for test set:')
#print((actuals_predicts['actual'] == actuals_predicts['predicted']).value_counts())

