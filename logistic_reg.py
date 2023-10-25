import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#---- 1. LOADING
data = pd.read_csv('data/train.csv')

#---- 2. CLEANING
data['Transported'] = data['Transported'].astype('int32')
data["IsNull_" + data.columns] = data.isna()

data = pd.concat([data, pd.DataFrame({"HomePlanet_Earth": data['HomePlanet'] == 'Earth',
                                      "HomePlanet_Europa": data['HomePlanet'] == 'Europa',
                                      "HomePlanet_Mars": data['HomePlanet'] == 'Mars'})], axis=1)
data = pd.concat([data, pd.DataFrame({"Destination_TRAPPIST": data['Destination'] == 'TRAPPIST-1e',
                                      "Destination_PSO": data['Destination'] == 'PSO J318.5-22',
                                      "Destination_Cancri": data['Destination'] == '55 Cancri e'})], axis=1)

data[['CryoSleep', 'VIP']] = data[['CryoSleep', 'VIP']].fillna(False)
data['Age'].fillna(0., inplace=True)
data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0.)

#---- 3. TRAIN-TEST SPLIT
data_train = data.sample(frac=.8, random_state=400)
data_test = data.drop(data_train.index.tolist())

#---- 4. FITTING
model = smf.logit(
    'Transported ~ HomePlanet_Earth + HomePlanet_Europa + CryoSleep + IsNull_CryoSleep + Destination_TRAPPIST + Destination_PSO + Age' \
    ' + RoomService + IsNull_RoomService + FoodCourt + ShoppingMall + Spa + VRDeck' \
    , data_train)
fit_results = model.fit()
print(fit_results.summary())
print(fit_results.pred_table())

#---- 5. PREDICTING
predictions = fit_results.predict(data_test.drop(columns='Transported'))
actuals_predicts = pd.DataFrame({'actual': data_test['Transported'], 'predicted': predictions}).dropna()
actuals_predicts['predicted'] = (actuals_predicts['predicted'] >= .5).astype('int32')

print('\nPrediction accuracy for test set:')
print((actuals_predicts['actual'] == actuals_predicts['predicted']).value_counts())