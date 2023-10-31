import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#---- 1. LOADING
main = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#---- 2. CLEANING
main['Transported'] = main['Transported'].astype('int32')

for df in [main, test]:
    df[["cabin_1", "cabin_2", "cabin_3"]] = df["Cabin"].str.split("/", expand=True)

    df['HomePlanet'].fillna(df['HomePlanet'].mode()[0], inplace=True)
    df['CryoSleep'].fillna(df['CryoSleep'].mode()[0], inplace=True)
    df['Destination'].fillna(df['Destination'].mode()[0], inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['VIP'].fillna(df['VIP'].mode()[0], inplace=True)
    df['RoomService'].fillna(df['RoomService'].mean(), inplace=True)
    df['FoodCourt'].fillna(df['FoodCourt'].mean(), inplace=True)
    df['ShoppingMall'].fillna(df['ShoppingMall'].mean(), inplace=True)
    df['Spa'].fillna(df['Spa'].mean(), inplace=True)
    df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=True)
    df['cabin_1'].fillna(df['cabin_1'].mode()[0], inplace=True)
    df['cabin_2'].fillna(df['cabin_2'].mode()[0], inplace=True)
    df['cabin_3'].fillna(df['cabin_3'].mode()[0], inplace=True)

    for deck in ["A", "B", "C", "D", "E", "F", "G"]:
        df[f"cabin_{deck}"] = (df['cabin_1'] == deck)
    df['cabin_2'] = df['cabin_2'].astype('int32')
    df['cabin_starboard'] = (df['cabin_3'] == 'S')

main = pd.concat([main, pd.DataFrame({"HomePlanet_Earth": main['HomePlanet'] == 'Earth',
                                    "HomePlanet_Europa": main['HomePlanet'] == 'Europa'})], axis=1)
main = pd.concat([main, pd.DataFrame({"Destination_TRAPPIST": main['Destination'] == 'TRAPPIST-1e',
                                    "Destination_PSO": main['Destination'] == 'PSO J318.5-22'})], axis=1)
test = pd.concat([test, pd.DataFrame({"HomePlanet_Earth": test['HomePlanet'] == 'Earth',
                                    "HomePlanet_Europa": test['HomePlanet'] == 'Europa'})], axis=1)
test = pd.concat([test, pd.DataFrame({"Destination_TRAPPIST": test['Destination'] == 'TRAPPIST-1e',
                                    "Destination_PSO": test['Destination'] == 'PSO J318.5-22'})], axis=1)

#---- 3. TRAIN-TEST SPLIT
main_train = main.sample(frac=.8, random_state=400)
main_test = main.drop(main_train.index.tolist())

#---- 4. FITTING
model = smf.logit(
    'Transported ~ HomePlanet_Earth + HomePlanet_Europa + CryoSleep + Destination_TRAPPIST + Destination_PSO + Age' \
    ' + RoomService + FoodCourt + ShoppingMall + Spa + VRDeck + cabin_starboard' \
    , main_train)
fit_results = model.fit()
print(fit_results.summary())
print(fit_results.pred_table())

#---- 5. PREDICTING
train_predicts = fit_results.predict(main_test.drop(columns='Transported'))
train_results = pd.DataFrame({'actual': main_test['Transported'], 'predicted': train_predicts})
train_results['predicted'] = (train_results['predicted'] >= .5).astype('int32')

print('\nPrediction accuracy for test set:')
print((train_results['actual'] == train_results['predicted']).value_counts())

test_predicts = fit_results.predict(test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": (test_predicts >= .5)})

submission.to_csv("submission.csv", index=False)