from rr_forest import RRForestClassifier
from rr_extra_forest import RRExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
import pandas as pd

datapath = '../data/'
fname = 'cs-training.csv'
data = pd.read_csv(datapath + fname, index_col=0)

# data_sto = data.sample(frac=1)
data_sto = data
Y_sto = data_sto['SeriousDlqin2yrs']
X_sto = data_sto.drop('SeriousDlqin2yrs', axis=1)
X_sto['MissingIncome'] = X_sto['MonthlyIncome'].isnull().apply(lambda b: int(b))

def replace_nan_minus1(x):
    if pd.isnull(x):
        return -1
    else:
        return x

X_sto['MonthlyIncome'] = X_sto['MonthlyIncome'].apply(replace_nan_minus1)

def replace_nan_0(x):
    if pd.isnull(x):
        return 0
    else:
        return x

X_sto['NumberOfDependents'] = X_sto['NumberOfDependents'].apply(replace_nan_0)

Xtest = pd.read_csv(datapath + 'cs-test.csv', index_col=0).drop('SeriousDlqin2yrs', axis=1)
Xtest['MissingIncome'] = Xtest['MonthlyIncome'].isnull().apply(lambda b: int(b))
Xtest['MonthlyIncome'] = Xtest['MonthlyIncome'].apply(replace_nan_minus1)
Xtest['NumberOfDependents'] = Xtest['NumberOfDependents'].apply(replace_nan_0)



clf = RRExtraTreesClassifier(n_estimators=20)
clf.fit(X_sto, Y_sto)


prediction = xclas.predict_proba(Xtest)
Xtest['Probability'] = prediction[:, 1]
Xtest['Id'] = Xtest.index
submission =Xtest[['Id', 'Probability']]

submission.to_csv('submission.csv')
submission.index.name = 'Id'
submission = submission.drop('Id', axis=1)
submission.to_csv('submission.csv')
