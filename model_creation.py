## Helper functions
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def create_logistic_regression(the_data,dep_var, weight_var=None, cat_vars=None):
    # Rework categorical variables so that they're dummy 0/1 vars
    dat = the_data.copy(deep=True)
    if cat_vars != None:
        if type(cat_vars) is list:
            # Quick sanity check that the input was appropiate
            dat = pd.get_dummies(dat, columns=cat_vars,drop_first=True)
    
    X = dat.drop([dep_var],axis=1,inplace=False).to_numpy()
    X = preprocessing.scale(X) # This is not fully proper, I should be using the pipeline methodology
    y = dat[dep_var].to_numpy()
    if weight_var == None:
        log_model = LogisticRegression(random_state=1984, penalty='none',solver='lbfgs', max_iter=1000).fit(X, y)
    else:
        log_model = LogisticRegression(random_state=1984, penalty='none',solve='lbfgs', class_weight=weight_var).fit(X, y)
    return log_model
#End

def create_random_forest_model(the_data, dep_var, weight_var=None, cat_vars = None):
    dat = the_data.copy(deep=True)
    if cat_vars != None:
        if type(cat_vars) is list:
            # Quick sanity check that the input was appropiate
            dat = pd.get_dummies(dat, columns=cat_vars,drop_first=True)
    
    X = dat.drop([dep_var],axis=1,inplace=False).to_numpy()
    y = dat[dep_var].to_numpy()
    if weight_var == None:
        tree_model = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=100,
                                            min_samples_leaf=25, random_state=1984).fit(X,y)
    else:
        tree_model = 2
    return tree_model
#End

def create_naive_bayes_model(the_data, dep_var, weight_var=None,cat_vars=None):
    dat = the_data.copy(deep=True)
    if cat_vars != None:
        if type(cat_vars) is list:
            # Quick sanity check that the input was appropiate
            dat = pd.get_dummies(dat, columns=cat_vars,drop_first=True)
    
    X = dat.drop([dep_var],axis=1,inplace=False).to_numpy()
    y = dat[dep_var].to_numpy()
    if weight_var == None:
        naive_model = GaussianNB().fit(X,y)
    else:
        naive_model = 2
    return naive_model
#End