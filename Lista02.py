import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
import statsmodels.api as sm

def normalizar_serie(serie):
    minimo = min(serie)
    maximo = max(serie)
    y = (serie - minimo) / (maximo - minimo)
    return y

def gerar_janelas(tam_janela, serie):
    # serie: vetor do tipo numpy ou lista
    tam_serie = len(serie)
    tam_janela = tam_janela +1 # Adicionado mais um ponto para retornar o target na janela
    janela = list(serie[0:0+tam_janela]) #primeira janela p criar o objeto np
    janelas_np = np.array(np.transpose(janela))     
    for i in range(1, tam_serie-tam_janela):  #começa do 1 
        janela = list(serie[i:i+tam_janela])
        j_np = np.array(np.transpose(janela))        
        janelas_np = np.vstack((janelas_np, j_np)) 
    return janelas_np

def split_serie_with_lags(serie, perc_train, perc_val = 0):
    #faz corte na serie com as janelas já formadas 
    x_date = serie[:, 0:-1]
    y_date = serie[:, -1]          
    train_size = np.fix(len(serie) *perc_train)
    train_size = train_size.astype(int)
    if perc_val > 0:        
        val_size = np.fix(len(serie) *perc_val).astype(int)   
        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]
        print("Particao de Treinamento:", 0, train_size  )  
        x_val = x_date[train_size:train_size+val_size,:]
        y_val = y_date[train_size:train_size+val_size]
        print("Particao de Validacao:",train_size, train_size+val_size)
        x_test = x_date[(train_size+val_size):-1,:]
        y_test = y_date[(train_size+val_size):-1]
        print("Particao de Teste:", train_size+val_size, len(y_date))
        return x_train, y_train, x_test, y_test, x_val, y_val
    else:
        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]
        x_test = x_date[train_size:-1,:]
        y_test = y_date[train_size:-1]
        return x_train, y_train, x_test, y_test
    
def treinar_mlp(x_train, y_train, x_val, y_val, num_exec, func_ativacao):
    neuronios =  [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ] #[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 170, 200]
    func_activation = func_ativacao#['tanh', 'relu']   #['identity', 'tanh', 'relu']
    alg_treinamento = ['lbfgs', 'adam']#, 'sgd', 'adam']
    max_iteracoes = [1000] #[100, 1000, 10000]
    learning_rate = ['constant', 'adaptive']  #['constant', 'invscaling', 'adaptive']
    qtd_lags_sel = len(x_train[0])
    best_result = np.Inf
    for i in range(0,len(neuronios)):
        for j in range(0,len(func_activation)):
            for l in range(0,len(alg_treinamento)):
                for m in range(0,len(max_iteracoes)):
                    for n in range(0,len(learning_rate)):
                        for qtd_lag in range(1, len(x_train[0]+1)): #variar a qtd de pontos utilizados na janela 
#                            print('QTD de Lags:', qtd_lag, 'Qtd de Neuronios' ,neuronios[i], 'Func. Act', func_activation[j])
                            for e in range(0,num_exec):
                                mlp = MLPRegressor(hidden_layer_sizes=neuronios[i], activation=func_activation[j], solver=alg_treinamento[l], max_iter = max_iteracoes[m], learning_rate= learning_rate[n])
                                mlp.fit(x_train[:,-qtd_lag:], y_train)
                                predict_validation = mlp.predict(x_val[:,-qtd_lag:])
                                mse = MSE(y_val, predict_validation)
                                if mse < best_result:
                                    best_result = mse
#                                    print('Melhor MSE:', best_result)
                                    select_model = mlp
                                    qtd_lags_sel = qtd_lag
    return select_model, qtd_lags_sel

def retorna_mse_mlp(y_test, func_ativacao):
    modelo, lag_sel = treinar_mlp(x_train, y_train, x_val, y_val, 1, [func_ativacao])
    predict_test = modelo.predict(x_test[:, -lag_sel:])
    return MSE(y_test, predict_test)
# =============================================================================
# 01
# =============================================================================
dados = pd.read_csv('airline.txt', header=None)
plt.plot(dados)
plt.show()

serie = dados.values
serie_normalizada = normalizar_serie(serie)
plt.plot(serie_normalizada)
plt.show()

sm.graphics.tsa.plot_acf(serie_normalizada, lags=20)

tam_janela = 20
serie_janelas = gerar_janelas(tam_janela, serie_normalizada)
x_train, y_train, x_test, y_test, x_val, y_val = split_serie_with_lags(serie_janelas, 0.50, perc_val = 0.25)

list_mse_tanh = []
melhor_mse_tanh = np.Inf
for e in range(0, 0):
    print('Execução Teste TANH: ', e)
    mse = retorna_mse_mlp(y_test, 'tanh')
    list_mse_tanh.append(mse)
    if mse < melhor_mse_tanh:
        melhor_mse_tanh = mse

list_mse_relu = []
melhor_mse_relu = np.Inf
for e in range(0, 0):
    print('Execução Teste RELU: ', e)
    mse = retorna_mse_mlp(y_test, 'relu')
    list_mse_relu.append(mse)
    if mse < melhor_mse_relu:
        melhor_mse_relu = mse

print('Média TANH: ', np.mean(list_mse_tanh))
print('Desvio TANH: ', np.std(list_mse_tanh))
print('Melhor MSE TANH: ', melhor_mse_tanh)

print('Média RELU: ', np.mean(list_mse_relu))
print('Desvio RELU: ', np.std(list_mse_relu))
print('Melhor MSE RELU: ', melhor_mse_relu)

# =============================================================================
# 1h40min
# Média TANH:  0.0012618586697446907
# Desvio TANH:  0.0002921093782500738
# Melhor MSE TANH:  0.0009438369652282301
# Média RELU:  0.0013001808313116098
# Desvio RELU:  0.0005016446044085206
# Melhor MSE RELU:  0.0008109609855106403
# =============================================================================

# =============================================================================
# 02
# =============================================================================
from pandas_datareader import data
dados = data.DataReader('AAPL', start='2018', end='2019', data_source='yahoo') #Ações da Apple
dados.head()
serie = dados['Adj Close']
#plt.plot(serie)
#plt.show()
serie_normalizada = normalizar_serie(serie)
#plt.plot(serie_normalizada)
#plt.show()
tam_janela = 20
serie_janelas = gerar_janelas(tam_janela, serie_normalizada)
x_train, y_train, x_test, y_test, x_val, y_val = split_serie_with_lags(serie_janelas, 0.50, perc_val = 0.25)

modelo, lag_sel = treinar_mlp(x_train, y_train, x_val, y_val, 1, ['relu'])
predict_train = modelo.predict(x_train[:, -lag_sel:])
predict_val = modelo.predict(x_val[:, -lag_sel:])
predict_test = modelo.predict(x_test[:, -lag_sel:])

previsoes_train = np.hstack(( predict_train, predict_val))
target_train = np.hstack((y_train, y_val))

plt.plot(predict_train, label = 'Prev Tr + val')
plt.plot(target_train, label='Target')
plt.legend(loc='best')
plt.show()

plt.plot(predict_test, label = 'Prev Test')
plt.plot(y_test, label='Target')
plt.legend(loc='best')
plt.show()
print(MSE(y_test, predict_test))
# =============================================================================
# Melhor MSE 0.003943196706037042
# =============================================================================
        
import elm
#params = ["linear", 5, []]
elmk = elm.ELMKernel()
elmk.search_param(np.array(dados), cv="ts", of="mse", eval=10)
train_set, test_set = elm.split_sets(np.array(dados), training_percent=.75, perm=False)

tr_result = elmk.train(train_set)
te_result = elmk.test(test_set)

plt.plot(tr_result.predicted_targets, label = 'Previsão Treino ELM')
plt.plot(tr_result.expected_targets, label='Target ELM')
plt.legend(loc='best')
plt.show()

plt.plot(te_result.predicted_targets, label = 'Previsão Teste ELM')
plt.plot(te_result.expected_targets, label='Target ELM')
plt.legend(loc='best')
plt.show()
print(te_result.get_mse())
# =============================================================================
# Melhor MSE 0.5653698218594645
# =============================================================================
#Os dois modelos foram ambos executados 1 vez, com 75% para treino e validadção, e 25% para testes. A métrica utilizada foi a de MSE