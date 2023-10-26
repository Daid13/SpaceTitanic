import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#---- 1. LOADING
data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#---- 2. CLEANING
data['Transported'] = data['Transported'].astype('int32')

for x in [data, test]:
    x['HomePlanet'].fillna(x['HomePlanet'].mode()[0], inplace=True)
    x['CryoSleep'].fillna(x['CryoSleep'].mode()[0], inplace=True)
    x['Destination'].fillna(x['Destination'].mode()[0], inplace=True)
    x['Age'].fillna(x['Age'].mean(), inplace=True)
    x['VIP'].fillna(x['VIP'].mode()[0], inplace=True)
    x['RoomService'].fillna(x['RoomService'].mean(), inplace=True)
    x['FoodCourt'].fillna(x['FoodCourt'].mean(), inplace=True)
    x['ShoppingMall'].fillna(x['ShoppingMall'].mean(), inplace=True)
    x['Spa'].fillna(x['Spa'].mean(), inplace=True)
    x['VRDeck'].fillna(x['VRDeck'].mean(), inplace=True)

data = pd.concat([data, pd.DataFrame({"HomePlanet_Earth": data['HomePlanet'] == 'Earth',
                                    "HomePlanet_Europa": data['HomePlanet'] == 'Europa'})], axis=1)
data = pd.concat([data, pd.DataFrame({"Destination_TRAPPIST": data['Destination'] == 'TRAPPIST-1e',
                                    "Destination_PSO": data['Destination'] == 'PSO J318.5-22'})], axis=1)
test = pd.concat([test, pd.DataFrame({"HomePlanet_Earth": test['HomePlanet'] == 'Earth',
                                    "HomePlanet_Europa": test['HomePlanet'] == 'Europa'})], axis=1)
test = pd.concat([test, pd.DataFrame({"Destination_TRAPPIST": test['Destination'] == 'TRAPPIST-1e',
                                    "Destination_PSO": test['Destination'] == 'PSO J318.5-22'})], axis=1)

#---- 3. TRAIN-TEST SPLIT
data_train = data.sample(frac=.8, random_state=400)
data_test = data.drop(data_train.index.tolist())

#---- 4. FITTING
model = smf.logit(
    'Transported ~ HomePlanet_Earth + HomePlanet_Europa + CryoSleep + Destination_TRAPPIST + Destination_PSO + Age' \
    ' + RoomService + FoodCourt + ShoppingMall + Spa + VRDeck' \
    , data_train)
fit_results = model.fit()
print(fit_results.summary())
print(fit_results.pred_table())

#---- 5. PREDICTING
train_predicts = fit_results.predict(data_test.drop(columns='Transported'))
train_results = pd.DataFrame({'actual': data_test['Transported'], 'predicted': train_predicts})
train_results['predicted'] = (train_results['predicted'] >= .5).astype('int32')

print('\nPrediction accuracy for test set:')
print((train_results['actual'] == train_results['predicted']).value_counts())

test_predicts = fit_results.predict(test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": (test_predicts >= .5)})

submission.to_csv("submission.csv", index=False)