
# coding: utf-8

# In[6]:

import pandas as pd 
import collections as c 
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls 
from plotly.graph_objs import * 
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm 



#loads the data 
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')


#creates new column from the loansdata dataframe where we strip out the % sign at the end 
clean_interest_rate = loansData['Interest.Rate'].map(lambda x: x.rstrip('%'))


#converts data into a float 
clean_interest_rate = clean_interest_rate.map(lambda x: float(x))

#turns data into a decimal by dividing by 100 
clean_interest_rate = clean_interest_rate.map(lambda x: x / 100)

#ensures the decimals only round to 4 digits
clean_interest_rate = clean_interest_rate.map(lambda x: round(x, 4))

#defines the loansdata interest rate dataframe as the cleaned up version of it
loansData['Interest.Rate'] = clean_interest_rate


#calls a new column to clean up the loan length dataframe and removes the word months 
clean_loan_length = loansData['Loan.Length'].map(lambda x: x.rstrip(' months'))

#converts the data to an integer
clean_loan_length = clean_loan_length.map(lambda x: int(x))

#creates a new dataframe and sets them equal 
loansData['Loan.Length'] = clean_loan_length

#creates a new fico score column
loansData['FICO.Score'] = [int(x.split('-')[0]) for x in loansData['FICO.Range']]

#create a histogram of the fico score 
plt.figure()
loansData['FICO.Score'].hist()
plt.show()


#defining the variables 
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

#dependent variable 
y = np.matrix(intrate).transpose()

#independent variables shaped as columns 
x1 = np.matrix(loanamt).transpose()
x2 = np.matrix(fico).transpose()

x = np.column_stack([x1, x2])

x = sm.add_constant(x) 
model = sm.OLS(y, x)
f = model.fit()

print f.summary()

#this creates a new dataframe that checks whether the values in Interest Rate is greater or less than .12 
loansData['IR_TF'] = loansData['Interest.Rate'].map(lambda x: 1 if x > .12 else 0)

#this creates a new column with a 1 value, which is the intercept
loansData['intercept'] = loansData['IR_TF'].map(lambda x: 1 if x == 0 else 1)

#define intercept 
intercept = loansData['intercept']

#x3 is now defined as a transposed intercept
x3 = np.matrix(intercept).transpose()

#listing all the independent varliables
ind_vars = np.column_stack([x1, x2, x3])

#defining the logit function
logit = sm.Logit(loansData['IR_TF'], ind_vars)

#showing me the result 
result = logit.fit()

#printing out the result and the coefficients 
coeff = result.params
print coeff

#this prints the results of the logit function 
print result.summary()

#log function with x1 and x2 and then run it with the numbers 720 and 10000 for fico and loan amt 
def logistic_function(x1, x2):
	p = 1 /(1 + np.exp(-(coeff[2] + coeff[0]*x1 + coeff[1]*x2)))
	return p

print logistic_function(10000, 720)

loansData.to_csv('loansData_clean.csv', header=True, index=False)

