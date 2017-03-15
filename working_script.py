# coding: utf-8
import pandas as pd
get_ipython().magic('pinfo pd.read_csv')
import os
os.chdir('/Users/stan/PycharmProjects/CS274a/HW6')
data_path = os.path.join('data', 'dataset1.txt')
data
data_path
get_ipython().magic('pinfo pd.read_csv')
data_path1 = os.path.join('data', 'dataset1.txt')
data_path1
data1 = pd.read_csv(data_path1)
get_ipython().magic('pinfo data1.head')
data1.head()
get_ipython().magic('pinfo pd.read_csv')
data1 = pd.read_csv(data_path1, header = ['X1', 'X2'])
get_ipython().magic('pinfo pd.read_csv')
data1 = pd.read_csv(data_path1, header=None, names=['X1', 'X2'])
data1
data1 = pd.read_csv(data_path1, header=None)
data1
get_ipython().magic('pinfo pd.read_csv')
data1 = pd.read_csv(data_path1, header=None)
data1
data1.columns = ['X1', 'X2']
get_ipython().magic('pinfo data1.columns')
data1.columns
data1 = pd.read_csv(data_path1)
data1
data1 = pd.read_csv(data_path1, header=None)
data1
data1[0][0]
get_ipython().magic('pinfo pd.read_csv')
data1 = pd.read_csv(data_path1, header=None, names=['X1', 'X2'])
data1
data1 = pd.read_csv(data_path1, sep=' ', header=None, names=['X1', 'X2'])
data1
data1
data1[:][0]
data1[0:3][0]
data1[0][0]
data1[0][0]
data1
data1 = pd.read_csv(data_path1)
data1
data1 = pd.read_csv(data_path1, header=None)
data1
data1[0][0]
data1 = pd.read_csv(data_path1, header=None, sep=' ')
data1
data1 = pd.read_csv(data_path1, header=None, sep='+\s')
data1 = pd.read_csv(data_path1, header=None, sep='\s+')
data1
data1 = pd.read_csv(data_path1, header=None, sep='\s+', names=['X1', 'X2'])
data1
data1 = pd.read_csv(data_path1, header=None, sep='\s+', names=['X1', 'X2'])
data1
import matplotlib.pyplot as plt
plt.scatter(data1)
data1.plot(kind='scatter')
data1.plot(x='X1', y='X2'  kind='scatter')
plt.plot(x=data1[0], y=data1[1])
get_ipython().magic('pinfo plt.plot')
plt.plot(data1[0], data1[1])
x = data1[0]
x
data1[0]
data1[0][0]
data1
data1['X1']
plt.plot(data1['X1'], data1['X2'])
plt.show()
get_ipython().magic('pinfo plt.plot')
plt.scatter(data1['X1'], data1['X2'])
plt.show()
get_ipython().magic('pinfo plt.scatter')
plt.scatter(data1['X1'], data1['X2'], s=1)
plt.show()
data1
get_ipython().magic('save')
get_ipython().magic('pwd ')
get_ipython().magic('save script_1')
get_ipython().magic('pinfo %save')
get_ipython().magic('save filename script1')
os.chdir('/Users/stan/PycharmProjects/CS274a/HW6')
get_ipython().magic('pwd ')
get_ipython().magic('save working_script.py')
get_ipython().magic('save -a working_script.py')
get_ipython().magic('save working_script.py')
