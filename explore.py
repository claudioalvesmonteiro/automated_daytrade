'''
TRADE
'''

# 
#import the libraries
import pandas as pd
#download tick data for stock
#df = pd.read_csv("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=^BVSP&interval=5min&outputsize=full&apikey=UDCCZOE6N1IMGNER&datatype=csv")
df = pd.read_csv("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=^BVSP&outputsize=full&apikey=UDCCZOE6N1IMGNER&datatype=csv")
df = df[len(df)-1100:len(df)]
#=============================
# criar variaveis preditoras
#============================

#----------- identificar queda e subida
df['candle'] = [1 if df['close'][x] >= df['open'][x] else 0 for x in range(len(df))]

#------------ criar variavel 'variacao na compra anterior'

# loop para mensurar variavel
var_anterior = []
for i in range(len(df)):
    if i < len(df)-1:
        var_anterior.append(df['open'][i+1] / df['close'][i+1] )

# remover ultimo caso na base e adicionar variavel
df = df.drop(len(df)-1)
df['variacao_anterior'] = var_anterior

#------------ criar variavel 'soma ultimos 4 candles'

# loop para mensurar variavel
cont_candle1 = []
cont_candle2 = []
cont_candle3 = []
cont_candle4 = []
cont = 2
while cont <= 5:
    for i in range(len(df)):
        valor = 0 
        for soma in range(cont):
            if soma > 0:
                if i < 1880:
                    valor = valor + df['candle'][i+soma]
        if cont == 2:
            cont_candle1.append(valor)
        elif cont == 3:
            cont_candle2.append(valor)
        elif cont == 4:
            cont_candle3.append(valor)
        elif cont == 5:
            cont_candle4.append(valor)
    cont = cont+1

# adicionar colunas a base
df['cont_candle1'] = cont_candle1
df['cont_candle2'] = cont_candle2
df['cont_candle3'] = cont_candle3
df['cont_candle4'] = cont_candle4


#=============================
# RANDOM FORESTS
#============================

df.dropna(inplace=True)

# selecionar colunas
features = df[[ 'variacao_anterior', 
                'cont_candle1', 
                'cont_candle2', 
                'cont_candle3', 
                'cont_candle4']]

label = df['candle']

# separar treinamento teste
features_train = features[150:]
features_test = features[0:150]
label_train = label[150:]
label_test = label[0:150]

# importar modelo
from sklearn.ensemble import RandomForestRegressor

# carregar modelo com 1000 arvores
modelo = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
modelo = modelo.fit(features_train, label_train)


#=============================
# TESTE e AVALIACAO
#=============================

# probabilidades da previsao
probs = modelo.predict(features_test)
previsoes = [1 if x > 0.5 else 0 for x in probs]

# metricas de avaliacao
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# matrix de confusao
print(confusion_matrix(label_test, previsoes))

# metricas 
print(classification_report(label_test, previsoes))

# auc
fpr,tpr, threshold = roc_curve(label_test, probs)
roc_auc = auc(fpr, tpr)

# curva roc
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
