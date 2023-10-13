import pandas as pd
from matplotlib import pyplot as plt


def do_scatterplot(i, col):
    ax = f.add_subplot(5, 2, i)
    ax.boxplot([not_transported[col], transported[col]], vert=False)
    ax.set_yticklabels(["Still here", "Gone...."])
    ax.set_title(col, loc="right", y=0.6)

def do_bar(i, col):
    ax = f.add_subplot(5, 2, i)

    still_here, gone = 0, 0
    for val in df[col].sort_values().unique():
        d_still_here = not_transported[col].where(lambda x: x == val).count()
        d_gone = transported[col].where(lambda x: x == val).count()
        ax.barh([False, True],
           [d_still_here, d_gone],
           left = [still_here, gone]
        )
        still_here += d_still_here
        gone += d_gone

    ax.set_yticks([False, True])
    ax.set_yticklabels(["Still here", "Gone...."])
    ax.set_title(col, loc="right", y=0.6)
    ax.legend(df[col].sort_values().unique())


df = pd.read_csv("data/train.csv")
not_transported = df.where(lambda p: p["Transported"] == False).dropna()
transported = df.where(lambda p: p["Transported"] == True).dropna()
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
