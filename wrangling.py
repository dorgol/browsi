import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/dorgol/browsi/main/browsi_data.csv')
    df[["load time (ms)", "avg scroll depth in url"]] = df[["load time (ms)", "avg scroll depth in url"]]. \
        apply(pd.to_numeric)
    df_tmp = df.drop(['ad_id', 'site_id', 'site_domain', 'user_id', 'device_vendor'], axis=1)
    df_tmp = pd.get_dummies(df_tmp)
    df_tmp = df_tmp.sample(frac=1,axis=0)

    return df_tmp

def split(df):
    y = df.is_viewd
    X = df.drop('is_viewd', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27, shuffle=True,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def fitting_model(X, y, model, **kwargs):
    model_to_fit = model(**kwargs)
    return model_to_fit.fit(X,y)

def predict_report(fitted, x_test: pd.DataFrame, y_test):
    y_pred = fitted.predict(x_test)
    ac_score = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return y_pred, ac_score, confusion, report

def oversampling(X_data, y_data):
    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=27, shuffle=True)
    columns = X_train.columns
    os_data_X, os_data_y = os.fit_resample(X_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['is_viewd'])
    return os_data_X, os_data_y, X_test, y_test


df = pd.read_csv('https://raw.githubusercontent.com/dorgol/browsi/main/browsi_data.csv')

# for col in df.columns:
#     df[col] = df[col].astype('category')

df[["load time (ms)", "avg scroll depth in url"]] = df[["load time (ms)", "avg scroll depth in url"]].\
    apply(pd.to_numeric)


def plot_hist(col):
    fig = px.histogram(df, x=col)
    return fig

def plot_corr(col_a, col_b):
    col_a = df.loc[:,col_a]
    col_b = df.loc[:,col_b]
    co_mat = pd.crosstab(col_a, col_b)
    return px.imshow(co_mat)

def plot_pivot(indices, cols, agg_func='sum'):
    if agg_func == 'sum':
        table = pd.pivot_table(df, values='is_viewd', index=indices,
                               columns=cols, aggfunc=agg_func, fill_value=-0.1)
    else:
        table = pd.pivot_table(df, values='is_viewd', index=indices,
                               columns=cols, aggfunc=np.mean, fill_value=-0.1)
    return px.imshow(table, color_continuous_scale='viridis')

