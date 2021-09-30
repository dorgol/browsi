import pandas as pd
import streamlit as st
import numpy as np
import wrangling
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import RFE, RFECV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate


st.title('Viewing Predictor')
st.markdown('''In this report I will try to build a predictor to identify whether ads were viewed or not.\nI will show 
  the process of building and evaluating the classifier. I will also include steps that didn't improve the prediction.
  The code will be attached in a Github repository. ''')

st.subheader('Data Exploration')
st.markdown('''First, let's take a look at the data. We can see the first rows of the data:''')
st.write(wrangling.df.head())
st.markdown('''We can see that the data include information about the behavior of the user (time of reading, 
source of traffic, depth of scrolling etc) and information about the ad and device (measures of ad and screen, 
type of browser etc.). 
\n In the next table we can see the number of unique values in every column:''')
st.write(wrangling.df.nunique())
st.markdown('''We can see that there are 4065 ads in the data. There are 3786 users, 
so it is evident that most of the users appear only once. We can also see that there are about 300 sites spread 
across 15 different categories. From this table we can't tell what is the distribution of the variables and what 
are the relations between variables.
\nIn the next tab we can learn more about the distribution of each variable.''')

cols = st.selectbox('columns', ['hour', 'is_viewd', 'browser', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])

a = wrangling.plot_hist(cols)
st.plotly_chart(a)

st.markdown('''We can learn some patterns from the plots above. First, we can see that the data isn't balanced. 
It contains 685 positive instances out of 4065 (~16%). We will use this information later on.   
We can see that most of the activity happen between 4 a.m to 17 p.m, from a mobile, through Facebook ios, 
on a wifi connection. The dimension of the ad and screens are mostly constant.
 \nThese plots still doesn't reveal us the connections between variables. We can learn from such stories - 
 Do people use different devices on different hours? Do they view more ads? 
 Is there a connection between the screen and the ad dimension? 
 can we devide users to "persistent users" vs. "bypassers"? In order to get some sense out of these questions we can
 observe the next tab. We can group by and sum/mean variables.
 \n  
 ''')

col_a = st.selectbox('columns to select', ['hour', 'is_viewd', 'site_category', 'browser', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])

col_b = st.selectbox('columns to select 2', ['hour', 'is_viewd', 'site_category', 'browser', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])

b = wrangling.plot_corr(col_a, col_b)
st.plotly_chart(b)

st.markdown('''We can notice a few things; first, "screen width" and "ad height" are highly correlated, and so do 
"device type" and "screen width". It indicates that most of the data deals with mobile ads with 
fixed screens and hence fixed ad sizes. I don't see clear evidence for persistence.
\nNext we would like to see whether we can see any connections to the target variable. 
For that end we will look at the next tab. We can group by multiple variable where the color in each square shows the
mean or sum of views within the grouped variables.''')

indices = st.multiselect('group by indices', ['hour', 'is_viewd', 'browser', 'site_category', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])

cols = st.multiselect('group by indices2', ['hour', 'is_viewd', 'browser', 'site_category', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])


agg_func = st.selectbox('function', ['mean', 'sum'])
d = wrangling.plot_pivot(indices, cols, agg_func)
d

st.markdown('''We can see that there is a correlation between the structure of the ad and screen and its viewability. 
The sum and mean of "ad height" and "screen width" indicates that some sizes are more viewable. 
It appears that ads in apps are more viewable but that may be due to small sample. ''')


st.subheader('Data Processing')
st.markdown('''The basic data processing needed is label encoding. Many variables are categorical and can't be used 
as is in a model. Label encoding is needed in order to create new columns of indicators of categories. For example,
if a column contains 3 categories it will become 3 columns of zeros and ones, each indicating single category.
The ad_id can't be used as it is merely an id and doesn't contain any information. Right now user_id will also be 
omitted as it has almost no variance in it - every user visits 1.1 sites. We will also omit site_id for now, as 
it will make the data highly sparse. We will assume that most of the data is encapsulate within the site category.''')

st.subheader('Models')
st.markdown('''In this section I will fit models to the data. I will start with simple models, 
will try to find problems and improve the models accordingly. In the process I will show what was done regarding the
features and what classifiers were chosen. 
\nI will start with a simple logistic regression model as a basic model. The main benefit of the logistic 
regression model is its simplicity, albeit it can also be its challenge. Since we have decent amount of entries in
our data we will split it to train and test sets, we will also shuffle the data since it is has some consecutive
ones. We will use the stratify option to split the data to two similar distributions.''')

from experiment import load_data, split, fitting_model, predict_report, oversampling
from sklearn.linear_model import LogisticRegression

df = load_data()
X_train, X_test, y_train, y_test = split(df)
logit = fitting_model(X_train, y_train, LogisticRegression, solver = 'liblinear', max_iter = 200)
_, _, confusion, report = predict_report(logit, X_test, y_test)
st.markdown('#### Results - Basic Logistic Regression')
st.markdown('''We can now see the Results of the model. The first table is the confusion matrix, where we can 
 see the correct and misspecified predictions by the model for zeros and ones. The most prominent aspect of the matrix
 is the lack of predicted ones. The model predicts exclusively zeros. This is due to the imbalanced data - the include
 about 85% of zeros, and the rest are ones. It "rewards" the model to predict zeros.''')
st.write(pd.DataFrame(confusion))
st.markdown('''The next table is a general report on the classifier performance. The table reports on the precision 
within the positive and negative categories. This is the natural way of thinking about a classifier - does the 
predictions are correct or not? We can see a 83% precision rate in the negative class but a 0% in the positive class.
The report shows a plain average of these two numbers as "macro avg" (42%), and a weighted average as "weighted avg" 
(69%). The report shows the recall rates per class (100% and 0%). The recall indicated how much relevant instances
are retrieved by the model. f1 combines the two measures by using harmonic mean. 
\nThis model clearly performs poorly although it has 83% general precision. To avoid the misconception we can look at 
the poor f1 and recall scores within the positive instances. This indicates that this model fails colossally. 
The model can't separate between the positive and negative instances at all. 
In order to deal with this problem we will use our next trick.''')
st.text(report)

st.markdown('#### Results - SMOTE Logistic Regression')
st.markdown('''We will use Synthetic Minority Oversampling TEchnique, or SMOTE. This method is a common way to deal 
with imbalanced data. It creates fake minority points by finding close minority points and creating new ones between
the existing ones. We will create new points so that we will have equal number of zeros and ones. Naturally, we will
do it only with the train set. 
\nWe can now look at the confusion matrix of the model with the new synthetic data:''')

X_train, y_train, X_test, y_test = oversampling(df.drop('is_viewd', axis=1), df.is_viewd)
logit = fitting_model(X_train, y_train, LogisticRegression, solver = 'liblinear', max_iter = 200)
_, _, confusion, report = predict_report(logit, X_test, y_test)
st.write(pd.DataFrame(confusion))
st.markdown('''There are a few things to notice. First, as expected this models assign much more weight to the 
positive class. It's expected since we now have more positive instances in our data. Second, alongside with the
increased weight of the positive class more negative instances wrongly assigned to the positive class. The precision
of the model is now 70% in general, worse than our former model if we are looking only from the 
precision perspective. We can see different aspects in the report below:''')

st.text(report)
st.markdown('''We can see the improvement in the recall of the positive class and in the precision averages. 
We can also see some deterioration in other aspects of the classifier, this is due to the misclassification of 
negative and positive instances.''')

st.markdown('#### Feature Selection')
st.markdown('''The dataset includes 39 variables. Multiple variables can harm the classification since a variable can 
contain more noise than signal. We will try to improve the classification by selecting informative features. 
We will do that by using Recursive Feature Elimination (RFE). We will use it with cross validation. The model starts
with the entire dataset and prune the least helpful variable until an optimal set or predefined number is achieved.
The following plot shows the number of variables and rate of correctly classifications. We can see that the model
with cross validation achieves about 80% of precision (recall that the data is now balanced so random 
classifier will get 50%). The model gets ~80% on the test set as well.''')


rfe = fitting_model(X_train, y_train, RFECV, estimator = LogisticRegression(solver='liblinear', max_iter=200),
                    step=1, cv=5, scoring='f1')
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
st.pyplot(plt)
st.caption('''We can see that there is one fold that behaves oddly. I tried to re-shuffle the data but it didn\'t affect
the problem''')

st.markdown('#### Models with subset of features')
st.markdown('''Now that we have a subset of features we will train a few models. We will add to the logistic regression
a random forest and adaBoost. We will use random forest because it will allow us to increase the complexity of the 
model and capture more nonlinear relations in our data that can't be captured with logistic regression. 
We will also use adaBoost. It will allow the model to focus on more intricate cases, 
and since viewing can be rather idiosyncratic it can be helpful.''')

x_sub_train = X_train[X_train.columns[rfe.support_]]
x_sub_test = X_test[X_test.columns[rfe.support_]]

random_forest = fitting_model(X_train, y_train, RandomForestClassifier,n_estimators=100,
                              max_depth=10, random_state=0)
_, _, confusion, report = predict_report(random_forest, X_test, y_test)
st.write(pd.DataFrame(confusion))
st.text(report)

st.markdown('''We can see that the random forest model in itself doesn't do a very good job. We can try to boost 
its performance a bit by optimizing its parameters. We will use grid search with the following parameters:''')

param_grid = {
        'n_estimators': [200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [4,8],
        'criterion' :['gini', 'entropy']
    }

st.write(param_grid)
st.markdown('''With the optimal combination of the parameters we get the following results:''')

grid_rf = fitting_model(X_train, y_train, GridSearchCV,estimator = RandomForestClassifier(),
                        param_grid=param_grid, cv= 2)

_, _, confusion, report = predict_report(grid_rf, X_test, y_test)
st.write(pd.DataFrame(confusion))
st.text(report)

ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=100, random_state=0)

adaboost = fitting_model(X_train, y_train, AdaBoostClassifier)
st.markdown('''We can see that grid search indeed improved the results of random forest but it's still not great. 
\nSince most of the models can't get a good results on the positives even on balanced dataset
we will also try adaBoost. This model uses weak learners additively and focuses on "tough cases" that were not 
solved in the former learners. This model gives us the following results:''')
_, _, confusion, report = predict_report(adaboost, X_test, y_test)
st.write(pd.DataFrame(confusion))
st.text(report)

st.markdown('''AdaBoost doesn't seem to do a very good job either. As a last improvement we will try to use an
ensemble of models. In a case where each model captures different aspect of the data an ensemble of models can succeed 
where single models can. If there is consensus between models it will not improve the results. We will use the models
that we used thus far.''')

estimators = [
('rf', random_forest),
('ada', ada)
]

stack_models = fitting_model(X_train, y_train, StackingClassifier, estimators=estimators,
                             final_estimator=LogisticRegression())
_, _, confusion, report = predict_report(stack_models, X_test, y_test)
st.write(pd.DataFrame(confusion))
st.text(report)

st.markdown('''The models seemed to be aligned ''')

st.markdown('### Graph Embedding')
st.markdown('''In this part I will try to use the data about the sites themselves. Thus far we didn't use this data 
since it will create sparse data and will avoid using the models with new sites. I will suggest a method for
extracting dense features instead of the current sparse representation. Having said that, it probably will
not work without using more data and calibrating the model. This part can be seen as a preparation for more elaborated
work. \nThe idea is the following: the data includes information about users and sites. This structure 
can be seen as bipartite graph. It's an unweighted, undirected graph. On this graph we can implement the node2vec 
algorithm. We will not get into the details of the algorithm, we will simply say that it uses random walks on the graph
in order to represent every node in the graph as a vector; The goal of the process is to create similar vectors to 
nodes that share many walks. We will create 10-dimensional embeddings to the sites and join them as features to 
a logistic regression with SMOTE. We can see the confusion table and the report below.''')

site_emb = pd.read_csv('https://raw.githubusercontent.com/dorgol/browsi/main/browsi_emb.csv')

df = pd.read_csv('https://raw.githubusercontent.com/dorgol/browsi/main/browsi_data.csv')
df[["load time (ms)", "avg scroll depth in url"]] = df[["load time (ms)", "avg scroll depth in url"]]. \
    apply(pd.to_numeric)
sites = df.site_id
df = df.drop(['ad_id', 'site_domain','site_id', 'user_id', 'device_vendor'], axis=1)
df = pd.get_dummies(df)
df = df.join(sites)
df = pd.merge(df,site_emb, left_on ='site_id', right_on='emb0.1')
df = df.drop(['site_id', 'emb0.1'], axis=1)
df = df.sample(frac=1,axis=0)




X_train, y_train, X_test, y_test = oversampling(df.drop('is_viewd', axis=1), df.is_viewd)

scoring = {'accuracy': 'balanced_accuracy', 'F1': 'f1', 'auc': 'roc_auc', 'precision': 'precision', 'recall': 'recall'}

modelCV = LogisticRegression()

results = cross_validate(modelCV, X_train, y_train, cv=10, scoring=list(scoring.values()),
                         return_train_score=True)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc],
                                                                          -results['test_%s' % list(scoring.values())[sc]].
                                                                          mean()
                                   if list(scoring.values())[sc]=='neg_log_loss'
                                   else results['test_%s' % list(scoring.values())[sc]].mean(),
                                   results['test_%s' % list(scoring.values())[sc]].std()))


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
mod = modelCV.fit(X_train, y_train)
y_pred=mod.predict(X_test)
st.write(confusion_matrix(y_test,y_pred))
st.text(classification_report(y_test, y_pred))

st.markdown('## Summary')
st.markdown('''In this app we tried to predict viewability by using different models and data. We tried to deal with 
the imbalance by using SMOTE, and we estimated logistic regression, random forest, adaBoost and stack of these models.
We used grid search to find better parameters. We also suggested to use graph embedding in order to use data about 
sites. It is clear that there are many more things to try and calibrate. This process can be viewed as merely a 
blueprint.''')