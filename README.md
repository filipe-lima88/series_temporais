# Séries Temporais
Exemplos de utilização de Séries Temporais

1- Realize uma análise na série de vendas no varejo no estado de Pernambuco (vendas_varejo_pe). Descreva quais componentes estão na
série; se é uma série estacionária; e se existe correlação entre os pontos da série.

2- Encontre um modelo ARMA(p,q) que melhor modela a série de chuva(chuva_fortaleza.xls) e a série de mortes por arma de fogo (morte_armas_australia.xls) . Justifique a sua escolha e a utilização de cada procedimento. (Utilize os correlogramas para encontrar os valores de p e q, e o teste ADF e KPSS para verificar estacionariedade).

3- Aplique a metodologia de Box e Jenkins para realizar a previsão de uma série financeira (à sua escolha). Divida o conjunto de dados em 70% para treinamento (estimação) e 30% teste. Comente cada etapa da metodologia. 

01 - Reproduza e apresente os resultados obtido do seguinte experimento: 
Foi realizada uma comparação entre duas funções de ativação: Relu e Tangente Hiperbólica, com o objetivo de encontrar qual a função que
apresenta um melhor desempenho da MLP para previsão da série Airline.Para tal, foi utilizado gridsearch variando diferentes parâmetros, como mostado abaixo:

Parâmetro Valores
Qtd de Neurônios: [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ]
Algoritmo de Treinamento: Método Quasi-Newton, Adam
Num de iterações: 10000
Atualização da Taxa de aprendizagem: Adaptativa, Constante
Tamanho máximo da janela: 20 pontos

Cada combinação de parâmetro foi executada 5 vezes. Os lags da janela foram pré-selecionados através da utilização da função de autocorrelação, e a quantidade de pontos da janela foram selecionados através do
gridsearch. A série foi particionada em 3 conjuntos: 50% para Treinamento, 25% para Validação e 25% para Teste.
Abaixo é mostrado o resultado (MSE) no conjunto de Teste obtido para 30 execuções.

Tanh: Média(0.0014015) Desvio(0.00018991) Melhor MSE(0.00115333)
Relu: Média(0.0018173) Desvio(0.00091390) Melhor MSE(0.00107004)

02 – Realize uma comparação entre a MLP e a ELM para previsão de uma série a sua escolha. Descreva cada etapa do processo, apresente como foi realizada essa comparação e os resultados obtidos.
