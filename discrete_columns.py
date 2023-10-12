import pandas as pd

df = pd.read_csv("data/train.csv")

discrete_columns = ["HomePlanet", "CryoSleep","Destination", "VIP"]


total_pas=df.size
total_transported=df["Transported"].value_counts()
print(total_transported)

for col in discrete_columns:
    col_values=df[col].sort_values().unique()
    for value in col_values:
        print(col, value)
        temp_df=df[df[col]==value]
        print(temp_df["Transported"].value_counts("True"))
        

#iterate through relevant columms
#make subset based on value in relevant column
#show percentage transported
#
