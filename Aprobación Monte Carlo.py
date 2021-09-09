
#LIBRERIAS NECESARIAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import sem
from scipy.stats import norm
import seaborn as sns
import sklearn
from sklearn import datasets, linear_model
import statsmodels.api as sm


#LOAD FILES
df = pd.read_csv('/Volumes/GoogleDrive/My Drive/Python/AVON/Simulation/apr.csv')
df = aprcsv[['Date', 'Close']]

df['Date'] = pd.to_datetime(df['Date'])

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df.to_excel('df.xlsx')


time = pd.read_csv('/Volumes/GoogleDrive/My Drive/Python/AVON/Simulation/time.csv')
time = timecsv
time2 = time2csv

#DATA FRAMES CON FECHAS SEPARADAS POR DIA, SERIE CONTINUA
cols=["year","month","day"]
time['date'] = time[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
time = time.drop("Unnamed: 0", axis = 1)
time = time.drop(["year", "month", "day"], axis = 1)
time = time.set_index('date')
time.to_csv('time.csv')

#DATA FRAME PARA SUAVIZAMIENTO DE SERIE CONTINUA
time2  = time
time2 = time2['Close'].rolling(3).mean()
time2 = time2.drop(time2.index[range(2)])
time2 = pd.DataFrame(time2)
time2.to_csv('time2.csv')

plt.figure(figsize =(8,5))
plt.plot(time['Close'], ls = '-', lw = 2, color = 'grey', label = 'Serie Continua')
plt.plot(time2['Close'], ls = '--', lw = 2, color = 'red', label = 'Suavizamiento')
plt.legend()
plt.show


#PLOT DE TODA LA SERIE DE ORACULUS
sns.regplot(x=df["Date"], y=df["Close"], 
            scatter_kws={"color": "grey"},
            line_kws={"color":"r","alpha":0.7,"lw":5})
plt.xlim(43300,44425)
plt.show()

#REGRESION OLS
mod = sm.OLS(df.Close, df.Date)
res = mod.fit()
print(res.summary())

#TRATAMIENTO PARA LA SERIE DE TIEMPO
#HOJA DE CSV (TIME)

dft = aprcsv[['Date', 'Close']]

import datetime as dt
dft['Date'] = pd.to_datetime(dft['Date'])

dft['year'] = dft['Date'].dt.year
dft['month'] = dft['Date'].dt.month
dft.to_excel('dft.xlsx')

df = pd.read_csv('/Users/pabloaguirresolana/Desktop/apr1.csv')

dft = pd.read_csv('/Volumes/GoogleDrive/My Drive/Python/AVON/Simulation/dft.csv')
dft = dftcsv
dft = dft.set_index('Date')

pp2 = pp['Close'].rolling(3).mean()
pp2 = pp2.drop(pp.index[range(2)])
pp2 = pd.DataFrame(pp2)
pp2.to_csv('pp2.csv')

plt.figure(figsize =(8,5))
plt.plot(pp['Close'], ls = '-', lw = 2, color = 'grey', label = 'Promedio Ponderado')
plt.plot(pp2['Close'], ls = '--', lw = 2, color = 'red', label = 'Suavizamiento')
plt.legend()
plt.show

df = dft2





---------------------------------------------
#SELECCIÓN DE DATA FRAMES PARA SIMULACIONES 
#PARA LAS SIMULACIONES PONER DF AL DATA FRAME SELECCIONADO
#EN ORDEN, NO CORRE TODAS JUNTAS, UNA POR UNA:

time = timecsv
time2 = time2csv

pp = ppcsv

time = time.set_index('date') 
time2 = time2.set_index('date')   

pp = pp.set_index('Date')
pp2 = pp.set_index('Date')


df = time 
df = time2

df = pp
df = pp2

 
--------------------------------------------
#SIMULACIONES MONTE CARLO 

#CAMBIOS PORCENTUALES PARA CADA OBSERVACION
dfPctChange = df.pct_change()
dfLogReturns = np.log(1 + dfPctChange)

#CHART CON LAS VARIACIONES PORCENTUALES
plt.figure(figsize =(10,5))
plt.plot(dfLogReturns)
plt.show

#ECUACIÓN BROWNIANA MOTION PARTE FIJA (DRIFT)
MeanLogReturns = np.array(dfLogReturns.mean())
VarLogReturns = np.array(dfLogReturns.var())
StdLogReturns = np.array(dfLogReturns.std())
Drift = MeanLogReturns - (0.5*VarLogReturns)
print('Drift =', Drift)

#INTERVALOS E ITERACIONES DEPENDE DE LA n DE CADA DATA FRAME
NumIntervals = 141
Iterations = 1000

#SEED PARA REPLICAR EL MODELO
np.random.seed(7)

#ECUACIÓN BROWNIANA PARTE ESTOCTASTICA
SBMotion = norm.ppf(np.random.rand(NumIntervals, Iterations))
DailyReturns =np.exp(Drift + StdLogReturns*SBMotion)
StartClose = df.iloc[0]
Closer = np.zeros_like(DailyReturns)
Closer[0] = StartClose

#SIMULACIONES MONTE CARLO
for t in range(1, NumIntervals):
    Closer[t] = Closer[t - 1] * DailyReturns[t]


#CHART CON TODAS LAS SIMULACIONES
plt.figure(figsize=(10,5))
plt.plot(Closer)   
AMZNTrend = np.array(df.iloc[:, 0:1])
plt.plot(AMZNTrend, ls = '-', lw = 5, color = 'black', label = 'Serie Original')   
plt.ylim(30,100)
plt.legend()
plt.show()


#DATA FRAME CON TODAS LAS SIMULACIONES 
#PARA SACAR DESCRIPITIVOS VARIAS LA n SEGUN DATA FRAME
pred = pd.DataFrame(Closer)
pred_mean = pred.loc[: , 0:1000].mean()
pred_mean.describe()

pred_mean.median()


#INTERVALOS DE CONFIANZA
sample = pred_mean

confidence_level = 0.95
degrees_freedom = sample.size - 1
sample_mean = np.mean(sample)
sample_standard_error = scipy.stats.sem(sample)

confidence_interval = scipy.stats.t.interval(confidence_level, 
                                             degrees_freedom, 
                                             sample_mean, 
                                             sample_standard_error)
print(confidence_interval)


#INTERVALOS DE CONFIANZA MEDIANA

sample = pred_mean

confidence_level = 0.95
degrees_freedom = sample.size - 1
sample_mean = np.median(sample)
sample_standard_error = scipy.stats.sem(sample)

confidence_interval = scipy.stats.t.interval(confidence_level, 
                                             degrees_freedom, 
                                             sample_mean, 
                                             sample_standard_error)
print(confidence_interval)









pred_mean.plot.box(vert = True, sym = '',
                                 boxprops = dict(linestyle='-', linewidth=1, color='black'),
                                 whiskerprops = dict(color='black'),
                                 medianprops = dict(linestyle='-.', linewidth=1, color='red')) 

pred_mean.plot.box()

















