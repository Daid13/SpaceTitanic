import pandas as pd
from matplotlib import pyplot as plt


def do_scatterplot(i, col):
    ax = f.add_subplot(5, 2, i)
    ax.boxplot([not_transported[col].dropna(), transported[col].dropna()], vert=False)
    ax.set_yticklabels(["Still here", "Gone...."] if i%2 != 0 else [])
    ax.set_title(col, loc="right", y=0.6)
    ax.set_ylim(0, 4)

def do_bar(i, col):
    ax = f.add_subplot(5, 2, i)

    valid_values = list(df[col].dropna().sort_values().unique())
    still_here, gone = 0, 0
    for val in valid_values:
        d_still_here = not_transported[col][not_transported[col] == val].count()
        d_gone = transported[col][transported[col] == val].count()
        ax.barh([False, True], [d_still_here, d_gone], left = [still_here, gone])
        still_here += d_still_here
        gone += d_gone
    ax.barh([False, True],
            [len(not_transported['VIP'][not_transported['VIP'].isna() == True]),
                len(transported['VIP'][transported['VIP'].isna() == True])],
            left=[still_here, gone])

    ax.set_yticks([False, True])
    ax.set_yticklabels(["Still here", "Gone...."] if i%2 != 0 else [])
    ax.set_title(col, loc="right", y=0.6)
    ax.legend(valid_values + ["NaN"], loc = 'upper left', fontsize = 'x-small')
    ax.set_ylim(-1, 3)


df = pd.read_csv("data/train.csv")
not_transported = df[df['Transported'] == False]
transported = df[df['Transported'] == True]
graph_type = {
    "PassengerId": "nope",
    "HomePlanet": "bar",
    "CryoSleep": "bar",
    "Cabin": "nope",
    "Destination": "bar",
    "Age": "scatter",
    "VIP": "bar",
    "RoomService": "scatter",
    "FoodCourt": "scatter",
    "ShoppingMall": "scatter",
    "Spa": "scatter",
    "VRDeck": "scatter",
    "Name": "nope",
    "Transported": "nope",
}

print(df.columns)
for col in df.columns:
    print(col, df[col].sort_values().unique())

f = plt.figure()
i = 0
for col in df.columns:
    match graph_type[col]:
        case "scatter":
            i += 1
            do_scatterplot(i, col)
        case "bar":
            i += 1
            do_bar(i, col)
        case _:
            pass

f.savefig("temp.png")
