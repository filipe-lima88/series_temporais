import numpy as np
import pandas as pd
import matplotlib.pylab
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import statsmodels.api as sm

#def mape(y_true, y_pred):
#    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mape(actual: np.ndarray, predicted: np.ndarray):

    actual = list(actual)
    predicted = list(predicted)
    n = len(actual)
    soma = 0
    for i in range(0, len(actual)):
        if actual[i] > 0.0:
            x = np.abs((predicted[i] - actual[i]) / actual[i])
        else:
            x = 0
        
        soma = x + soma
    return 100/n * soma

def acf_pacf(x, qtd_lag):
    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(221)
    fig = sm.graphics.tsa.plot_acf(x, lags=qtd_lag, ax=ax1)
    ax2 = fig.add_subplot(222)
    fig = sm.graphics.tsa.plot_pacf(x, lags=qtd_lag, ax=ax2)
    plt.show()

from statsmodels.tsa.stattools import adfuller, kpss
def teste_df(serie):
    #H0: série não estacionária 
    dftest = adfuller(serie, autolag='AIC')
    df_output = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags','Nº Observações'])
    for key,value in dftest[4].items():
        df_output['Valor Crítico (%s)'%key] = value
    print(df_output)
    if df_output[0] < df_output['Valor Crítico (5%)']: # descarta a H0 se o teste estatistico for menor que o valor crítico
        print('estacionária')
    else:
        print(df_output[0])
        print('não estacionária')

def teste_kpss(serie):
    #H0: série é estacionária 
    kptest = kpss(serie, regression='c')
    kp_output = pd.Series(kptest[0:3], index=['Test Statistic','p-value','#Lags'])
    for key,value in kptest[3].items():
        kp_output['Valor Crítico (%s)'%key] = value
    print(kp_output)
    if kp_output[0] > kp_output['Valor Crítico (5%)']: # descarta a H0 se o teste estatistico for MAIOR que o valor crítico 
        print('não estacionária')
    else:
        print(kp_output[0])
        print('estacionária')

# =============================================================================
# 01
# =============================================================================
dados = pd.read_excel('vendas_varejo_pe.xlsx')
dados.index = pd.date_range('1/1/2000', periods=189, freq='M', normalize =True)
del dados['data']
serie = dados['Venda']
plt.plot(serie)
plt.show() 
qtd_lag = 48
acf_pacf(serie, qtd_lag)
# =============================================================================
# Tendência e Sazonalidade
# Não estacionária
# Existe correlação entre os 5 primeiros pontos da série através do PACF. Ao plotar o gráfico com 10 lags isso fica bem claro
# =============================================================================

# =============================================================================
# 02
# =============================================================================
dados = pd.read_excel('chuva_fortaleza.xlsx')
dados.index = pd.date_range('1/1/1850', periods=130, freq='Y', normalize =True)
del dados['Year']
serie = dados['Milimitros']
plt.plot(serie)
plt.show()
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(serie) 
qtd_lag = 48
acf_pacf(serie, qtd_lag)
# =============================================================================
# de acordo com o correlograma, p e q = 1
# =============================================================================
teste_df(serie)
teste_kpss(serie)
# =============================================================================
# estacionária
# =============================================================================
tam_treinamento = int(len(serie) * 0.75)
dados_treinamento, dados_teste = serie.iloc[0:tam_treinamento].values, serie.iloc[tam_treinamento:].values
p = 2
d = 0
q = 1
# =============================================================================
# p,q 1 => aic 1473.404 pelo acf e pacf
# p 2 e q 1 => aic 1468.522 
# =============================================================================
modelo_arma = ARIMA(dados_treinamento, order=(p,d,q)).fit()
dados_prev = modelo_arma.fittedvalues 
plt.plot(dados_prev, label = 'ARMA train')
plt.plot(dados_treinamento, label='Real train')
plt.legend(loc='best')
plt.show()
#print(MSE(dados_treinamento, dados_prev))
print(modelo_arma.summary())

historico = [x for x in dados_treinamento]
previsoes = []
for i in range(len(dados_teste)):
    modelo = ARIMA(historico, order=(p,d,q)).fit()
    prev = modelo.forecast()[0]
    previsoes.append(prev)
    obs = dados_teste[i]
    historico.append(obs)
    
plt.plot(previsoes, label = 'ARMA')
plt.plot(dados_teste, label='Real')
plt.legend(loc='best')
plt.show()
print(MSE(dados_teste, previsoes))

dados = pd.read_excel('morte_armas_australia.xlsx')
dados.index = pd.date_range('1/1/1915', periods=90, freq='Y', normalize =True)
del dados['Year']
serie = dados['taxa_mortes'].diff().dropna()
# =============================================================================
# tive q acrescetar o diff().dropna() para que a serie passasse a ser estacionaria pelo teste_df()
# =============================================================================
plt.plot(serie)
plt.show()
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(serie) 
qtd_lag = 48
acf_pacf(serie, qtd_lag)
# =============================================================================
# de acordo com o correlograma p e q = 2, 4 antes do diff().dropna(). Após p e q = 1 
# =============================================================================
teste_df(serie) # estacionaria
teste_kpss(serie) # estacionaria
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error as MSE
tam_treinamento = int(len(serie) * 0.75)
dados_treinamento, dados_teste = serie.iloc[0:tam_treinamento].values, serie.iloc[tam_treinamento:].values
# =============================================================================
# melhor config p=1,d e q=0
# =============================================================================
#p = 1
#d = 0
#q = 0
# =============================================================================
# p = 0
# d = 0
# q = 1
# AIC                            -83.304
# 0.012482709415963151
# 
# p = 1
# d = 0
# q = 0
# AIC                            -75.052
# 0.012748691668496747
# 
# p = 1
# d = 0
# q = 1
# AIC                            -82.172
# 0.0130442694726727370
# =============================================================================
modelo_arma = ARIMA(dados_treinamento, order=(p,d,q)).fit()
dados_prev = modelo_arma.fittedvalues 
plt.plot(dados_prev, label = 'ARMA train')
plt.plot(dados_treinamento, label='Real train')
plt.legend(loc='best')
plt.show()
#print(MSE(dados_treinamento, dados_prev))
print(modelo_arma.summary())

historico = [x for x in dados_treinamento]
previsoes = []
for i in range(len(dados_teste)):
    modelo = ARIMA(historico, order=(p,d,q)).fit()
    prev = modelo.forecast()[0]
    previsoes.append(prev)
    obs = dados_teste[i]
    historico.append(obs)
    
plt.plot(previsoes, label = 'ARMA')
plt.plot(dados_teste, label='Real')
plt.legend(loc='best')
plt.show()
print(MSE(dados_teste, previsoes))

# =============================================================================
# 03
# =============================================================================
#from pandas.tseries.offsets import BDay
from pandas_datareader import data
dados = data.DataReader('AAPL', start='2018', end='2019', data_source='yahoo') #Ações da Apple
dados.head()
#dados.index = pd.date_range('2/1/2018', periods=252, freq=BDay(), normalize =True)
serie = dados['Adj Close']
plt.plot(serie)
qtd_lag = 40
acf_pacf(serie, qtd_lag)
# =============================================================================
# Identificação: Por se tratar de uma série não estacionária e possuir características de randomwalk, e não possuir sazionalidade, foi escolhido o modelo ARIMA.
# Através do FAC e FACP, foram escolhidos os valores p=1, d=2 e q=1.
# =============================================================================
p = 2
d = 0
q = 0
# =============================================================================
# Estimação: Separação dos dados para treinamento e teste em 70/30
# =============================================================================
tam_treinamento = int(len(serie) * 0.70)
dados_treinamento, dados_teste = serie.iloc[0:tam_treinamento].values, serie.iloc[tam_treinamento:].values
plt.plot(dados_treinamento, label='dados_treinamento')
plt.plot(dados_teste, label='dados_teste')
plt.legend(loc='best')
plt.show()
modelo = ARIMA(dados_treinamento, order=(p,d,q)).fit()
train_prev = modelo.fittedvalues
plt.plot(dados_treinamento, label='dados_treinamento')
plt.plot(train_prev, label='train_prev')
plt.legend(loc='best')
plt.show()
# =============================================================================
# Avaliação: Através do AIC, o modelo que possuir o menor score deve ser "considerado" como o melhor modelo
# Se o modelo estiver representando bem a série temporal, os resíduos devem ser um ruído branco; 
# =============================================================================
print(modelo.summary())
acf_pacf(modelo.resid, qtd_lag)
# =============================================================================
# Previsão: Aqui, o modelo escolhido e treinado, é aplicado para prever os dados da "base" de testes que foi separada na etapa de Separação
# O melhor modelo é escolhido através dos menores scores de MSE, MAE e MAPE
# =============================================================================
historico = [x for x in dados_treinamento]
previsoes = []
for i in range(len(dados_teste)):
    modelo = ARIMA(historico, order=(p,d,q)).fit()
    prev = modelo.forecast()[0]
    previsoes.append(prev)
    obs = dados_teste[i]
    historico.append(obs)
    
plt.plot(previsoes, label = 'ARIMA')
plt.plot(dados_teste, label='Real')
plt.legend(loc='best')
plt.show()
print(MSE(dados_teste, previsoes))
print(MAE(dados_teste, previsoes))
print(mape(dados_teste, previsoes))
# =============================================================================
#  Conclusão dos testes:
#  ARMA(1,0)
#  p = 1
#  d = 0
#  q = 0
#  AIC 839.495
#  MSE 21.285101184804397  -> MENOR MSE ENTRE OS PARAMETROS
#  MAE 3.6456271639995514
#  MAPE [1.90041678]
#
#  ARIMA(0,1,0)
#  p = 0
#  d = 1
#  q = 0
#  AIC 826.718
#  MSE 21.651660380142438  
#  MAE 3.6314396385763517  -> MENOR MAE ENTRE OS PARAMETROS
#  MAPE [1.89415074]       -> MENOR MAPE ENTRE OS PARAMETROS
# 
#  ARIMA(1,1,0) 
#  p = 1
#  d = 1
#  q = 0
#  AIC 825.359             -> MENOR AIC ENTRE OS PARAMETROS
#  MSE 22.09263070184655
#  MAE 3.66921663473125
#  MAPE [1.91286142]
#  
#  ARIMA(1,2,1)
#  p = 1
#  d = 2
#  q = 1
#  AIC 827.534
#  MSE 22.152846967215613
#  MAE 3.6566000357170885
#  MAPE [1.90385031]
# 
#  ARIMA(2,2,2) 
#  p = 2
#  d = 2
#  q = 2
#  AIC 829.524
#  MSE 22.350522947003515
#  MAE 3.679611947713313
#  MAPE [1.91673353]
# =============================================================================
