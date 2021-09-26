import streamlit as st
import numpy as np
import wrangling


st.title('Viewing Predictor')

st.write(wrangling.df.head())
st.write(wrangling.df.nunique())

cols = st.selectbox('columns', ['hour', 'is_viewd', 'browser', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])
a = wrangling.plot_hist(cols)
st.plotly_chart(a)

col_a = st.selectbox('columns to select', ['hour', 'is_viewd', 'browser', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])

col_b = st.selectbox('columns to select 2', ['hour', 'is_viewd', 'browser', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])

b = wrangling.plot_corr(col_a, col_b)
st.plotly_chart(b)

indices = st.multiselect('group by indices', ['hour', 'browser', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])

cols = st.multiselect('group by indices2', ['hour', 'browser', 'fold', 'device_type', 'connection',
                                'screen width', 'ad height', 'load time (ms)', 'attempt index',
                                'avg scroll depth in url', 'traffic_source'])


agg_func = st.selectbox('function', ['mean', 'sum'])
d = wrangling.plot_pivot(indices, cols, agg_func)
st.plotly_chart(d)
