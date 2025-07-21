import pandas as pd

# reading my file
df = pd.read_csv("C:/Users/pegahg/Desktop/LoanDefaultPrediction/Loan_default.csv")

# take a look at the data
print("Data shape:", df.shape)
print("first rows:\n", df.head())

#if there is any null cells
print("null colomns:\n", df.isnull().sum().sort_values(ascending=False))

#check the percentage of the Y
print ("tabe hadaf:\n", df['Default'].value_counts(normalize=True))

#split the data to X & Y
X = df.drop('Default', axis=1)
y = df['Default']

# check for data types cuz we need to encode the strings
print("Data types count:")
print(X.dtypes.value_counts())
X = X.drop(['LoanID'], axis=1, errors='ignore')
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"{col}: {X[col].nunique()}")

# encode
X = pd.get_dummies(X)
print("Shape after encoding:", X.shape)

# check the encoded file
X = X.astype(int)
X['Default'] = y
X.to_csv("C:/Users/pegahg/Desktop/LoanDefaultPrediction/loan_data_encoded.csv", index=False)

