import pandas as pd


# data cleaning 
df = pd.read_csv("penguins_size.csv")

# drop rows with no values...
df = df.dropna()
# drop duplecates
#df = df.drop_duplicates() there is no duplicates...

df['sex'] = df['sex'].replace({'MALE': 0, 'FEMALE': 1})

df['island'] = df['island'].replace({'Biscoe': 0, 'Dream': 1, 'Torgersen': 2})


print(df.head())

