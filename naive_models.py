import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
import wrangling
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.metrics import classification_report


#prepearing data
df = wrangling.df
df_tmp = df.drop(['ad_id','site_id', 'site_domain', 'site_category', 'user_id', 'device_vendor'], axis=1)
df_tmp = pd.get_dummies(df_tmp)
# df_tmp2 = df[['site_id', 'site_domain', 'site_category', 'user_id', 'device_vendor']]
# df = pd.concat([df_tmp, df_tmp2], axis=1)
y = df_tmp.is_viewd
X = df_tmp.drop('is_viewd', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

def logit(data_x, data_y, test_x, test_y, create_report = True):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
    pipe = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
    pipe.fit(data_x, data_y)
    y_pred = pipe.predict(test_x)
    if create_report == True:
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(pipe.score(X_test, test_y)))
        confusion_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
        print(confusion_matrix)
        print(classification_report(test_y, y_pred))

    return y_pred

logit(X_train, y_train, X_test, y_test)






#oversampling
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['is_viewd'])







clf = neighbors.KNeighborsClassifier(5, weights='distance')
clf.fit(X, y)
cl = clf.predict(X_test)
print(classification_report(y_test, cl))

logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()