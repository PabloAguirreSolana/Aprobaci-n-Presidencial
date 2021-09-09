#SIMULACIONES MONTE CARLO PARA APROBACIÓN PRESIDENCIAL RUTINA

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

#CARGAR ARCHIVOS

#SERIE CONTINUA
time = timecsv
time2 = time2csv

time = time.set_index('date') 
time2 = time2.set_index('date')   

#PROMEDIO PONDERADO
pp = ppcsv
pp2 = pp2csv

pp = pp.set_index('Date')
pp2 = pp2.set_index('Date')


df = time 
df = time2

df = pp
df = pp2

------------------------------------------------------------

#SIMULACIONES MONTE CARLO 

#CAMBIOS PORCENTUALES PARA CADA OBSERVACION
dfPctChange = df.pct_change()
dfLogReturns = np.log(1 + dfPctChange)


#ECUACIÓN BROWNIANA MOTION PARTE FIJA (DRIFT)
MeanLogReturns = np.array(dfLogReturns.mean())
VarLogReturns = np.array(dfLogReturns.var())
StdLogReturns = np.array(dfLogReturns.std())
Drift = MeanLogReturns - (0.5*VarLogReturns)
print('Drift =', Drift)

----------------------------------------------------------------
#100
#INTERVALOS E ITERACIONES DEPENDE DE LA n DE CADA DATA FRAME
NumIntervals = 141
Iterations = 100
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
#DATA FRAME CON TODAS LAS SIMULACIONES 
#PARA SACAR DESCRIPITIVOS E INTERVALOS DE CONFIANZA
pred = pd.DataFrame(Closer)
pred_meana = pred.loc[: , 0:100].mean()
summary1 = pred_meana.describe()
mad = pred_meana.mad()
sample = pred_meana
confidence_level = 0.95
degrees_freedom = sample.size - 1
sample_median = np.median(sample)
sample_standard_error = scipy.stats.sem(sample)
confidence_interval = scipy.stats.t.interval(confidence_level, 
                                             degrees_freedom, 
                                             sample_median, 
                                             sample_standard_error)
#RESULTADOS POR SIMULACION
print('confidence interval=', confidence_interval)
print('summary= ', summary1)
print('mad=', mad)
a = pd.DataFrame(pred_meana)
a.columns = ['value']
group = ['A']
a['group'] = np.repeat('A', 100)




#1000
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
#DATA FRAME CON TODAS LAS SIMULACIONES 
#PARA SACAR DESCRIPITIVOS E INTERVALOS DE CONFIANZA
pred = pd.DataFrame(Closer)
pred_meanb = pred.loc[: , 0:1000].mean()
summary1 = pred_meanb.describe()
mad = pred_meanb.mad()
sample = pred_meanb
confidence_level = 0.95
degrees_freedom = sample.size - 1
sample_median = np.median(sample)
sample_standard_error = scipy.stats.sem(sample)
confidence_interval = scipy.stats.t.interval(confidence_level, 
                                             degrees_freedom, 
                                             sample_median, 
                                             sample_standard_error)
#RESULTADOS POR SIMULACION
print('confidence interval=', confidence_interval)
print('summary= ', summary1)
print('mad=', mad)
b = pd.DataFrame(pred_meanb)
b.columns = ['value']
group = ['B']
b['group'] = np.repeat('B', 1000)



#10000
#INTERVALOS E ITERACIONES DEPENDE DE LA n DE CADA DATA FRAME
NumIntervals = 141
Iterations = 10000
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
#DATA FRAME CON TODAS LAS SIMULACIONES 
#PARA SACAR DESCRIPITIVOS E INTERVALOS DE CONFIANZA
pred = pd.DataFrame(Closer)
pred_meanc = pred.loc[: , 0:10000].mean()
summary1 = pred_meanc.describe()
mad = pred_meanc.mad()
sample = pred_meanc
confidence_level = 0.95
degrees_freedom = sample.size - 1
sample_median = np.median(sample)
sample_standard_error = scipy.stats.sem(sample)
confidence_interval = scipy.stats.t.interval(confidence_level, 
                                             degrees_freedom, 
                                             sample_median, 
                                             sample_standard_error)
#RESULTADOS POR SIMULACION
print('confidence interval=', confidence_interval)
print('summary= ', summary1)
print('mad=', mad)
c = pd.DataFrame(pred_meanc)
c.columns = ['value']
group = ['C']
c['group'] = np.repeat('C', 10000)


#PARA BOX PLOTS

boxtime =a.append(b).append(c)
sns.boxplot(x='group', y='value', data=boxtime, showfliers = False, palette = 'Greys')
plt.show()





boxtime.plot.box(vert = True, sym = '',
                                 boxprops = dict(linestyle='-', linewidth=1, color='black'),
                                 whiskerprops = dict(color='black'),
                                 medianprops = dict(linestyle='-.', linewidth=1, color='red'))




e = pd.DataFrame({ 'group' : np.repeat('D',100), 'value': np.random.uniform(12, size=100) })



sample = df
confidence_level = 0.95
degrees_freedom = sample.size - 1
sample_mean = np.mean(sample)
sample_standard_error = scipy.stats.sem(sample)
confidence_interval = scipy.stats.t.interval(confidence_level, 
                                             degrees_freedom, 
                                             sample_mean, 
                                             sample_standard_error)

print(confidence_interval)

































#CHART CON TODAS LAS SIMULACIONES
plt.figure(figsize=(10,5))
plt.plot(Closer)   
AMZNTrend = np.array(df.iloc[:, 0:1])
plt.plot(AMZNTrend, ls = '-', lw = 5, color = 'black', label = 'Serie Original')   
plt.ylim(30,100)
plt.legend()
plt.show()














