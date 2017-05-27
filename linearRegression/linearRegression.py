# quandl.com for datasets 50 sets per day for free
# logic bug in code somewhere. Forecast isn't days into the future...


import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# preprocessing to keep the features within -1, 1
# cross_validation split up the data to get rid of biases
# svm used to show how simple it is to change the algorithm
quandl.ApiConfig.api_key = 'zyGphgipxRGeSSAzFLLz'

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

#           price        x          x              x
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#set forecast_col variable here in case we use a different data set in the future (unrelated to stock prices)
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#predicting day in the future using 1% of the dataframe
forecast_out = int(math.ceil(0.1*len(df)))
# print(forecast_out)
#shifting df up 1% days into the future
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
#scale data before sending it to preprocesser(adds processing time, would skip this step if doing HFT)
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# ^shuffles up Xs and ys

clf = LinearRegression(n_jobs=-1)
# clf = LinearRegression(n_jobs=-1) run as many threads as possible
# clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)

#Serializing our data with pickle
    #pickled data is saved now so we can comment this code out and use a readonly from the pickle file so as to not generate the classifier ever time.
    #(SCALING ^ with pickle)

# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)
    #what are we dumping? clf, where? f

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

#train and test on different data otherwise the program already knows the answer
accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 #seconds per day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
