import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#---- 1. LOADING
data = pd.read_csv('data/train.csv')

#---- 2. CLEANING
data['Transported'] = data['Transported'].astype('int32')

data['HomePlanet'].fillna(data['HomePlanet'].mode()[0], inplace=True)
data['CryoSleep'].fillna(data['CryoSleep'].mode()[0], inplace=True)
data['Destination'].fillna(data['Destination'].mode()[0], inplace=True)
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['VIP'].fillna(data['VIP'].mode()[0], inplace=True)
data['RoomService'].fillna(data['RoomService'].mean(), inplace=True)
data['FoodCourt'].fillna(data['FoodCourt'].mean(), inplace=True)
data['ShoppingMall'].fillna(data['ShoppingMall'].mean(), inplace=True)
data['Spa'].fillna(data['Spa'].mean(), inplace=True)
data['VRDeck'].fillna(data['VRDeck'].mean(), inplace=True)

data = pd.concat([data, pd.DataFrame({"HomePlanet_Earth": data['HomePlanet'] == 'Earth',
                                      "HomePlanet_Europa": data['HomePlanet'] == 'Europa'})], axis=1)
data = pd.concat([data, pd.DataFrame({"Destination_TRAPPIST": data['Destination'] == 'TRAPPIST-1e',
                                      "Destination_PSO": data['Destination'] == 'PSO J318.5-22'})], axis=1)

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
predictions = fit_results.predict(data_test.drop(columns='Transported'))
actuals_predicts = pd.DataFrame({'actual': data_test['Transported'], 'predicted': predictions})
actuals_predicts['predicted'] = (actuals_predicts['predicted'] >= .5).astype('int32')

print('\nPrediction accuracy for test set:')
print((actuals_predicts['actual'] == actuals_predicts['predicted']).value_counts())