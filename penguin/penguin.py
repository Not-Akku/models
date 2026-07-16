import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# data cleaning 
# treat common placeholders as NA so dropna() removes them
df = pd.read_csv("penguin/penguins_size.csv", na_values=['NA', '.'])

# drop rows with no values...
df = df.dropna()
# drop duplecates
#df = df.drop_duplicates() there is no duplicates...

# changing the string values to numeric...
df['sex'] = df['sex'].replace({'MALE': 0, 'FEMALE': 1})

df['island'] = df['island'].replace({'Biscoe': 0,
                                     'Dream': 1,
                                     'Torgersen': 2})



df['species'] = df['species'].replace({'Adelie': 0,
                                       'Chinstrap': 1,
                                       'Gentoo': 2})

# ensure mapped columns are numeric (dropna() removed missing rows)
df['sex'] = df['sex'].astype(int)
df['island'] = df['island'].astype(int)
df['species'] = df['species'].astype(int)


# features for this model...
# features for this model (force numeric dtypes)
x = df.drop(columns='species').astype(float)
# target..
y = df['species']

# split the dataset to train and test by 20%
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2, random_state= 23)
# prepration of model
#kn = KNeighborsClassifier(n_neighbors=3)

#kn.fit(x_train, y_train)
model = DecisionTreeClassifier(max_depth= 3)
model.fit(x_train, y_train)



# predicting values
y_predict = model.predict(x_test)
# check the score of the model for futher changes and optimization...
score = accuracy_score(y_test, y_predict)

print("Accuracy:", score)

print(df.head())