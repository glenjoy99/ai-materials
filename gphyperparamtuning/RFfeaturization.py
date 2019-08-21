
# coding: utf-8

# In[1]:


print("Using Matminer Featurization to Run and Optimize Scikit Learn Models")
from matminer.utils.conversions import str_to_composition
from matminer.featurizers.composition import ElementProperty
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Viewing Dataset through Panda
pd.read_csv('../datasets/HydrogenStorageDataBasemagpie.data.csv')


# In[3]:


# Imporing and storing dataset into DataFrame object
df= pd.read_csv('../datasets/HydrogenStorageDataBasemagpie.data.csv')
# Allowing composition names in the dataset to be in  a recognizable format 
df["composition"] = df["comp"].transform(str_to_composition)
# Applying Magpie preset attributes for each composition
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id="composition")
# Setting target output value to Enthalpy
y = df['Enthalpy'].values
# Exluding our output value and all non-numeric values from our input
excluded = ["Enthalpy", "Phase", "comp", "composition"]
X = df.drop(excluded, axis =1)


# In[4]:


df.head()
print("There are {} possible descriptors:\n\n{}".format(X.shape[1], X.columns.values))


# In[5]:


# Importing Random Forest, MSE, and numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


# In[6]:


# Initializing a Random Forest Regressor and fitting it to our data
rf = RandomForestRegressor()
rf.fit(X, y)


# In[7]:


# Printing training stats
print('training R2 = ' + str(round(rf.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X))))


# In[8]:


# Importing and setting up 10-Fold Cross Validation
from sklearn.model_selection import KFold, cross_val_score
crossvalidation = KFold(n_splits=10, shuffle=False, random_state=1)
scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)


# In[9]:


# Printing Validation Results
print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[10]:


# Importing libraries for plotting
from matminer.figrecipes.plot import PlotlyFig
from sklearn.model_selection import cross_val_predict


# In[12]:


# Creates and plots experimental vs. predicted enthalpies
pf = PlotlyFig(x_title='Experimental Enthalpy',
               y_title='Predicted Enthalpy',
               title='Random Forest Regressor',
               mode='notebook',
               filename="rf_regression.html")

pf.xy(xy_pairs=[(y, cross_val_predict(rf, X, y, cv=crossvalidation)), ([0, 200], [0, 200])], 
      labels=df['comp'], 
      modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], 
      showlegends=False
     )

