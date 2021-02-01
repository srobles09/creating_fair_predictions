# German Credit Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
import sys
sys.path.append("D:/Sandy Oaks/Documents/Grad School/F20_MATH-5027/Code!/creating_fair_predictions/")
import discovery_metrics as dm
import model_creation as mc

path ='D:/Sandy Oaks/Documents/Grad School/F20_MATH-5027/'
data_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
## Metrics to analyze:
# Theil's Index    
# Difference of Means (not aif)
# Difference in proportions (not aif)
# Demographic/statistical Parity difference
# Equal opportunity difference
# Average Odds Difference
# Disparate Impact
# Impact Ratio (slift)


##---------------------- Read census income Data ----------------------##
## Read German Credit Data
column_names = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit'] # as given by german.doc

the_data = pd.read_csv(data_path, sep=' ', header=None, names=column_names,na_values=[])
status_map = {'A91': 'Male', 'A93': 'Male', 'A94': 'Male',
                  'A92': 'Female', 'A95': 'Female'} #They split it out by marital status, so it needed to be recombined
the_data['gender'] = the_data['personal_status'].replace(status_map)
the_data.drop(['personal_status'],axis=1,inplace=True)
the_data['credit'] = np.where(the_data['credit'] == 1, 1, 0)
the_data.dropna(inplace=True)

##---------------------- Investigate the data pre-modeling ----------------------##
dat_plot = the_data.copy()
dat_plot['credit'] = np.where(dat_plot['credit'] == 1, "Good Credit Risk", "Bad Credit Risk")
dat_plot['Age'] = pd.cut(dat_plot.age,bins=[9,19,29,39,49,59,69,79,89,99,10000],labels=['10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99','100+'])
sbn.countplot(y='Age', hue='credit', data=dat_plot,palette='muted')
sbn.countplot(y='gender', hue='credit', data=dat_plot,palette='muted')
plt.legend(loc='lower right')
del dat_plot


##---------------------- Data Cleaning ----------------------##
## Data manipulation and assignmentt of meta-variables
the_data['gender'] = np.where(the_data['gender'] == 'Male', 1, 0)
the_data['age'] = np.where(the_data['age'] > 25, 1, 0)
(data_dev,data_val) = train_test_split(the_data, train_size=0.75, random_state=1984)

dep_var = 'credit' # List of independent variables
# WALDO drop personal_status
Independent = ['status', 'month', 'credit_history', 'purpose', 'credit_amount',
       'savings', 'employment', 'investment_as_income_percentage',
       'other_debtors', 'residence_since', 'property', 'age',
       'installment_plans', 'housing', 'number_of_credits', 'skill_level',
       'people_liable_for', 'foreign_worker', 'gender'] # List of independent variables to go into model
to_subset =['status', 'month', 'credit_history', 'purpose', 'credit_amount',
       'savings', 'employment', 'investment_as_income_percentage',
       'other_debtors', 'residence_since', 'property', 'age',
       'installment_plans', 'housing', 'number_of_credits', 'skill_level',
       'people_liable_for', 'foreign_worker', 'credit', 'gender'] # This one is Independent + dep_var
#weight_var = weights = {0:3.0, 1:1.0} # Class of 0 gets a weight of 3


cat_vars = ['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level',
                     'foreign_worker']
num_vars = []
unpriv_gender = [{'gender': 0}]
priv_gender =[{'gender': 1}]
unpriv_age = [{'age': 0}]
priv_age =[{'age': 1}]
unpriv_both = [{'gender': 0,'age':0}]
priv_both =[{'gender': 1, 'age':1}]








##---------------------- Pre-model Bias Detection ----------------------##
data_dev = data_dev[to_subset] #subset to only the vars to be used in modeling
data_val = data_val[to_subset] #subset to only the vars to be used in modeling

### For those dependent on aif360
# Create binarylabeldataset object
dat_aif = data_dev.copy(deep=True)
dat_aif_val = data_val.copy(deep=True)
dat_aif = pd.get_dummies(dat_aif, columns=cat_vars,drop_first=True)
dat_aif_val = pd.get_dummies(dat_aif_val, columns=cat_vars,drop_first=True)

# Dev
dat_aif_gender = BinaryLabelDataset(df=dat_aif,
                                  label_names=['credit'],
                                  protected_attribute_names=['gender'],
                                  favorable_label=1,
                                  unfavorable_label=0,
                                  unprivileged_protected_attributes=0)

dat_aif_age = BinaryLabelDataset(df=dat_aif,
                                  label_names=['credit'],
                                  protected_attribute_names=['age'],
                                  favorable_label=1,
                                  unfavorable_label=0,
                                  unprivileged_protected_attributes=0)
dat_aif_both = BinaryLabelDataset(df=dat_aif,
                                  label_names=['credit'],
                                  protected_attribute_names=['gender','age'],
                                  favorable_label=1,
                                  unfavorable_label=0,
                                  unprivileged_protected_attributes=[0,0])

# Val
dat_aif_val_gender = BinaryLabelDataset(df=dat_aif_val,
                                  label_names=['credit'],
                                  protected_attribute_names=['gender'],
                                  favorable_label=1,
                                  unfavorable_label=0,
                                  unprivileged_protected_attributes=0)

dat_aif_val_age = BinaryLabelDataset(df=dat_aif_val,
                                  label_names=['credit'],
                                  protected_attribute_names=['age'],
                                  favorable_label=1,
                                  unfavorable_label=0,
                                  unprivileged_protected_attributes=0)
dat_aif_val_both = BinaryLabelDataset(df=dat_aif_val,
                                  label_names=['credit'],
                                  protected_attribute_names=['gender','age'],
                                  favorable_label=1,
                                  unfavorable_label=0,
                                  unprivileged_protected_attributes=[0,0])


# Create pre-processing metric object
data_prepropcessing_metrics_gender = BinaryLabelDatasetMetric(
        dataset=dat_aif_gender,
        unprivileged_groups=unpriv_gender,
        privileged_groups=priv_gender)
data_prepropcessing_metrics_age = BinaryLabelDatasetMetric(
        dataset=dat_aif_age,
        unprivileged_groups=unpriv_age,
        privileged_groups=priv_age)
data_prepropcessing_metrics_both = BinaryLabelDatasetMetric(
        dataset=dat_aif_both,
        unprivileged_groups=unpriv_both,
        privileged_groups=priv_both)

# Explainer for pre-processing metrics (for reference)
data_explainer_gender = MetricTextExplainer(data_prepropcessing_metrics_gender)
data_explainer_age = MetricTextExplainer(data_prepropcessing_metrics_age)
data_explainer_both = MetricTextExplainer(data_prepropcessing_metrics_both)

# Print out the desired metrics (used for paper)
data_prepropcessing_metrics_gender.statistical_parity_difference()
data_prepropcessing_metrics_age.statistical_parity_difference()
data_prepropcessing_metrics_both.statistical_parity_difference()

data_prepropcessing_metrics_gender.consistency(n_neighbors=10)
data_prepropcessing_metrics_age.consistency(n_neighbors=10)
data_prepropcessing_metrics_both.consistency(n_neighbors=10)

data_prepropcessing_metrics_gender.disparate_impact()
data_prepropcessing_metrics_age.disparate_impact()
data_prepropcessing_metrics_both.disparate_impact()


### Not dependent on aif360

# Difference of means test:
data1_age = the_data.credit[the_data.age==1] # priviledged (over 25)
data2_age = the_data.credit[the_data.age==0] # unpriviledged

data1_gender = the_data.credit[the_data.gender==1] # priviledged, male
data2_gender = the_data.credit[the_data.gender==0] # unpriviledged

data1_both = the_data.credit[(the_data.gender==1) & (the_data.age==1)] # priviledged, male age over 25
data2_both = the_data.credit[(the_data.gender==0) & (the_data.age==0)] # unpriviledged

stat_age, p_age = dm.get_difference_means_test(data1_age, data2_age)
stat_gender, p_gender = dm.get_difference_means_test(data1_gender, data2_gender)
stat_both, p_both = dm.get_difference_means_test(data1_both, data2_both)

##---------------------- Create models ----------------------##
# Correction has not been carried out at this stage
lr_model = mc.create_logistic_regression(data_dev,dep_var, cat_vars=cat_vars)
rf_model = mc.create_random_forest_model(data_dev,dep_var, cat_vars=cat_vars)
nb_model = mc.create_naive_bayes_model(data_dev,dep_var, cat_vars=cat_vars)


##---------------------- Post-model Bias Detection ----------------------##
# Create predictions
(X_scaled,y) = mc.clean_the_data(data_val[to_subset],dep_var, cat_vars, scale_me=True)
(X_unscaled,y) = mc.clean_the_data(data_val[to_subset],dep_var, cat_vars, scale_me=False)
del y # Using an internal function for the data cleaning, which outputs extraneous info (y)
pred_lr = lr_model.predict(X_scaled)
pred_rf = rf_model.predict(X_unscaled)
pred_nb = nb_model.predict(X_unscaled)



##---------------------- Correction ----------------------##


accuracy_score(data_val[dep_var],pred_lr)
accuracy_score(data_val[dep_var],pred_rf)
accuracy_score(data_val[dep_var],pred_nb)
