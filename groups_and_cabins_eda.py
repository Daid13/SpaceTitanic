import pandas as pd

df = pd.read_csv('data/train.csv')

_matchCabins = df['Cabin'].str.match('[A-Z]/\\d{,4}/[A-Z]')
_matchGroups = df['PassengerId'].str.match('\\d{4}_\\d{2}')
print('Cabin matches A/0/A:', _matchCabins[_matchCabins == True].count(), _matchCabins[_matchCabins == False].count(), len(_matchCabins))
print('PassengerId matches 0000_00:', _matchGroups[_matchGroups == True].count(), _matchGroups[_matchGroups == False].count(), len(_matchGroups))

_cabins = df['Cabin'].str.split('/', expand=True)
_groups = df['PassengerId'].str.split('_', expand=True)
df[['cabin_1', 'cabin_2', 'cabin_3']] = _cabins
df[['group_1', 'group_2']] = _groups
print(df)