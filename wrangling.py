import pandas as pd
import plotly.express as px
import numpy as np

df = pd.read_csv('C:/Users/dorgo/Downloads/browsi_data.csv')

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
    return px.imshow(table)

