import pandas as pd
from matplotlib import pyplot as plt


def do_scatterplot(i, col):
    ax = f.add_subplot(7, 2, i)
    ax.boxplot([nottransported[col], transported[col]], vert=False)
    ax.set_yticklabels(["Still here", "Gone...."])
    ax.set_title(col, loc="right", y=0.6)


df = pd.read_csv("data/train.csv")
nottransported = df.where(lambda p: p["Transported"] == False).dropna()
transported = df.where(lambda p: p["Transported"] == True).dropna()
graph_type = {
    "PassengerId": "nope",
    "HomePlanet": "nope",
    "CryoSleep": "scatter",
    "Cabin": "nope",
    "Destination": "nope",
    "Age": "scatter",
    "VIP": "scatter",
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
            print(i)
            do_scatterplot(i, col)
        case _:
            pass

f.savefig("temp.png")
