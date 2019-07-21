
# coding: utf-8

# # Logistic Regression

# Logistic regression is a statistical method for predicting binary classes. The outcome or target variable is dichotomous in nature. Dichotomous means there are only two possible classes. For example, it can be used for cancer detection problems. It computes the probability of an event occurrence.
# 
# It is a special case of linear regression where the target variable is categorical in nature. It uses a log of odds as the dependent variable. Logistic Regression predicts the probability of occurrence of a binary event utilizing a logit function.
# 
# Linear Regression Equation:

# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281880/image1_ga8gze.png)

# Sigmoid Function

# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281880/image3_qldafx.png)

# The last equation is called the Logistic Equation which is responsible for the calculation of Logistic Regression

# ### Linear Regression Vs. Logistic Regression

# Linear regression gives you a continuous output, but logistic regression provides a constant output. An example of the continuous output is house price and stock price. Example's of the discrete output is predicting whether a patient has cancer or not, predicting whether the customer will churn. Linear regression is estimated using Ordinary Least Squares (OLS) while logistic regression is estimated using Maximum Likelihood Estimation (MLE) approach.

# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281070/linear_vs_logistic_regression_edxw03.png)

# ### Sigmoid Function
# 
# The sigmoid function, also called logistic function gives an ‘S’ shaped curve that can take any real-valued number and map it into a value between 0 and 1. If the curve goes to positive infinity, y predicted will become 1, and if the curve goes to negative infinity, y predicted will become 0. If the output of the sigmoid function is more than 0.5, we can classify the outcome as 1 or YES, and if it is less than 0.5, we can classify it as 0 or NO. The outputcannotFor example: If the output is 0.75, we can say in terms of probability as: There is a 75 percent chance that patient will suffer from cancer.

# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281880/image4_gw5mmv.png) 

# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1534281070/sigmoid2_lv4c7i.png)

# ### class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

# 
# penalty : str, ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’)
# 
#     Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.
# 
#     New in version 0.19: l1 penalty with SAGA solver (allowing ‘multinomial’ + L1)
# dual : bool, optional (default=False)
# 
#     Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
# tol : float, optional (default=1e-4)
# 
#     Tolerance for stopping criteria.
# C : float, optional (default=1.0)
# 
#     Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
# fit_intercept : bool, optional (default=True)
# 
#     Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
# intercept_scaling : float, optional (default=1)
# 
#     Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight.
# 
#     Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.
# class_weight : dict or ‘balanced’, optional (default=None)
# 
#     Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
# 
#     The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
# 
#     Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
# 
#     New in version 0.17: class_weight=’balanced’
# random_state : int, RandomState instance or None, optional (default=None)
# 
#     The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random. Used when solver == ‘sag’ or ‘liblinear’.
# solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’).
# 
#     Algorithm to use in the optimization problem.
# 
#         For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
#         For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
#         ‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty
#         ‘liblinear’ and ‘saga’ also handle L1 penalty
#         ‘saga’ also supports ‘elasticnet’ penalty
#         ‘liblinear’ does not handle no penalty
# 
#     Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
# 
#     New in version 0.17: Stochastic Average Gradient descent solver.
# 
#     New in version 0.19: SAGA solver.
# 
#     Changed in version 0.20: Default will change from ‘liblinear’ to ‘lbfgs’ in 0.22.
# max_iter : int, optional (default=100)
# 
#     Maximum number of iterations taken for the solvers to converge.
# multi_class : str, {‘ovr’, ‘multinomial’, ‘auto’}, optional (default=’ovr’)
# 
#     If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.
# 
#     New in version 0.18: Stochastic Average Gradient descent solver for ‘multinomial’ case.
# 
#     Changed in version 0.20: Default will change from ‘ovr’ to ‘auto’ in 0.22.
# verbose : int, optional (default=0)
# 
#     For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
# warm_start : bool, optional (default=False)
# 
#     When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution. Useless for liblinear solver. See the Glossary.
# 
#     New in version 0.17: warm_start to support lbfgs, newton-cg, sag, saga solvers.
# n_jobs : int or None, optional (default=None)
# 
#     Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”. This parameter is ignored when the solver is set to ‘liblinear’ regardless of whether ‘multi_class’ is specified or not. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
# l1_ratio : float or None, optional (default=None)
# 
#     The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'`. Setting ``l1_ratio=0 is equivalent to using penalty='l2', while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2.
# 

# ## Simplified Implmentation

# In[147]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y)


# ## Implementation in Python

# ### Importing Libraries

# In[2]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Dataset

# In[3]:


link="D:/As a Trainer/Course Material/Machine Learning with Python/All Special/Logistic Regression/"
df = pd.read_csv(link+'diabetes.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# ### Exploratory Data Analysis

# In[7]:


sns.pairplot(df)


# In[8]:


df.isna()


# In[9]:


df[df.isna().any(axis=1)]


# In[13]:


df.isna().sum()


# ### Creating profile Report

# In[10]:


import pandas_profiling
pandas_profiling.ProfileReport(df)


# In[15]:


sns.distplot(df['BMI'],hist_kws=dict(edgecolor="black", linewidth=1),color='red')


# ### Correlation Plot

# In[16]:


df.corr()


# In[17]:


plt.figure(figsize=(8,8))
sns.heatmap(df.corr(), annot = True)


# In[21]:


sns.set_style('whitegrid')
sns.countplot(x='Outcome',hue='Outcome',data=df,palette='RdBu_r')


# ### Checking Distribution of Age for Diabates

# In[23]:


sns.distplot(df['Age'],kde=False,color='darkblue',bins=20)


# ### Splitting Dataset into Train and Test

# In[25]:


from sklearn.model_selection import train_test_split


# #### Taking Featured Columns

# In[34]:


X = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
y = ['Output']


# In[26]:


df2 = pd.DataFrame(data=df)
df2.head()


# #### Splitting

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome',axis=1),df['Outcome'],
                                                    test_size=0.30, random_state=101)


# ### Applying Logistic Regression

# In[29]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# #### Evaluation of Model

# In[30]:


predictions = logmodel.predict(X_test)


# In[31]:


print(predictions)


# In[32]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))


# ### Model Score

# In[43]:


X1 = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y1 = df['Outcome']


# In[44]:


from sklearn import metrics
print("Model Score: ",logmodel.score(X1,y1))


# ### Prediction section

# In[45]:


print("Please Enter Necessarry Values:\n")
[a,b,c,d,e,f,g,h]=eval(input('Pregnancies')),eval(input('Glucose')),eval(input('BloodPressure')),eval(input('SkinThickness')),eval(input('Insulin')),eval(input('BMI')),eval(input('DiabetesPedigreeFunction')),eval(input('Age'))


# In[48]:


pred=logmodel.predict([[a,b,c,d,e,f,g,h]])
if pred[0]==1:
    print("You have Duabetes!! Please consult with your doctor")
else:
    print("Congratulations!! You have eno Diabetes")


# ### Cross Validation

# In[56]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'
results = model_selection.cross_val_score(logmodel, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

