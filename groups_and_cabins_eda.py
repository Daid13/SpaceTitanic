import pandas as pd

df = pd.read_csv('data/train.csv')

_matchCabins = df['Cabin'].str.match('[A-Z]/\\d{,4}/[A-Z]')
print('Cabin matches A/0/A:', _matchCabins[_matchCabins == True].count(), _matchCabins[_matchCabins == False].count(), len(_matchCabins))

_cabins = df['Cabin'].str.split('/', expand=True)
df[['cabin_1', 'cabin_2', 'cabin_3']] = _cabins
print(df)