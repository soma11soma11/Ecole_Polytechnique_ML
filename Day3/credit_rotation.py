import pandas as pd
datapath = 'data/'
fname = 'cs-training.csv'
data = pd.read_csv(datapath + fname, index_col=0)

# data_sto = data.sample(frac=1)
data_sto = data
Y_sto = data_sto['SeriousDlqin2yrs']
X_sto = data_sto.drop('SeriousDlqin2yrs', axis=1)
X_sto['MissingIncome'] = X_sto['MonthlyIncome'].isnull().apply(lambda b: int(b))

def replace_nan_c(x, c):
    if pd.isnull(x):
        return c
    else:
        return x

X_sto['MonthlyIncome'] = X_sto['MonthlyIncome'].apply(lambda x: replace_nan_c(x, -1))
X_sto['NumberOfDependents'] = X_sto['NumberOfDependents'].apply(lambda x: replace_nan_c(x, 0))

Xtest = pd.read_csv(datapath + 'cs-test.csv', index_col=0).drop('SeriousDlqin2yrs', axis=1)
Xtest['MissingIncome'] = Xtest['MonthlyIncome'].isnull().apply(lambda b: int(b))
Xtest['MonthlyIncome'] = Xtest['MonthlyIncome'].apply(lambda x: replace_nan_c(x, -1))
Xtest['NumberOfDependents'] = Xtest['NumberOfDependents'].apply(lambda x: replace_nan_c(x, 0))

from rr_forest import RRForestClassifier

clf = RRForestClassifier(n_estimators=20)
clf.fit(X_sto.as_matrix(), Y_sto.as_matrix())

prediction = clf.predict_proba(Xtest.as_matrix())
Xtest['Probability'] = prediction[:, 1]
Xtest['Id'] = Xtest.index
submission1 =Xtest[['Id', 'Probability']]

submission1.index.name = 'Id'
submission1 = submission1.drop('Id', axis=1)
submission1.to_csv('submission1.csv')
