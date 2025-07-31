import pandas as pd 

columns_to_keep = ["path", "male"]

path = "maad_full_attributes.csv"

df = pd.read_csv(path)
print("here")
columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
print("here")
df.drop(columns=columns_to_drop, inplace=True)
print("here")
df.to_csv('maad_gender.csv', index=False)
print("here")