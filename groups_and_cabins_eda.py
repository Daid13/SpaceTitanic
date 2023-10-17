import pandas as pd
from matplotlib import pyplot as plt


def do_bar(i, col):
    ax = f.add_subplot(3, 2, i)

    valid_values = list(df[col].dropna().sort_values().unique())
    still_here, gone = 0, 0
    for val in valid_values:
        d_still_here = not_transported[col][not_transported[col] == val].count()
        d_gone = transported[col][transported[col] == val].count()
        ax.barh([False, True], [d_still_here, d_gone], left=[still_here, gone])
        still_here += d_still_here
        gone += d_gone
    ax.barh(
        [False, True],
        [
            len(not_transported["VIP"][not_transported["VIP"].isna() == True]),
            len(transported["VIP"][transported["VIP"].isna() == True]),
        ],
        left=[still_here, gone],
    )

    ax.set_yticks([False, True])
    ax.set_yticklabels(["Still here", "Gone...."] if i % 2 != 0 else [])
    ax.set_title(col, loc="right", y=0.6)
    ax.legend(valid_values + ["NaN"], loc="upper left", fontsize="x-small")
    ax.set_ylim(-1, 3)


df = pd.read_csv('data/train.csv')

_matchCabins = df['Cabin'].str.match('[A-Z]/\\d{,4}/[A-Z]')
_matchGroups = df['PassengerId'].str.match('\\d{4}_\\d{2}')
print('Cabin matches A/0/A:', _matchCabins[_matchCabins == True].count(), _matchCabins[_matchCabins == False].count(), len(_matchCabins))
print('PassengerId matches 0000_00:', _matchGroups[_matchGroups == True].count(), _matchGroups[_matchGroups == False].count(), len(_matchGroups))

_cabins = df['Cabin'].str.split('/', expand=True)
_groups = df['PassengerId'].str.split('_', expand=True)
df[['cabin_1', 'cabin_2', 'cabin_3']] = _cabins
df[['group_1', 'group_2']] = _groups
not_transported = df[df["Transported"] == False]
transported = df[df["Transported"] == True]
print(df)

f = plt.figure()
i = 0
for col in ['cabin_1', 'cabin_3', 'group_2']:
    i += 1
    do_bar(i, col)

f.savefig("groups_and_cabins.png")