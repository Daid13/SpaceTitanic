import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#---- 1. LOADING
data = pd.read_csv('data/train.csv')

#---- 2. CLEANING
data['Transported'] = data['Transported'].astype('int32')
data["IsNull_" + data.columns] = data.isna()

#---- 3. TRAIN-TEST SPLIT
data_train = data.sample(frac=.8, random_state=400)
data_test = data.drop(data_train.index.tolist())

#---- 4. FITTING
model = smf.logit('Transported ~ HomePlanet + CryoSleep + Destination + Age + VIP + RoomService + FoodCourt + ShoppingMall + Spa + VRDeck', data_train)
fit_results = model.fit()
print(fit_results.summary())
print(fit_results.pred_table())

#---- 5. PREDICTING
predictions = fit_results.predict(data_test.drop(columns='Transported'))
actuals_predicts = pd.DataFrame({'actual': data_test['Transported'], 'predicted': predictions}).dropna()
actuals_predicts['predicted'] = (actuals_predicts['predicted'] >= .5).astype('int32')

print('\nPrediction accuracy for test set:')
print((actuals_predicts['actual'] == actuals_predicts['predicted']).value_counts())