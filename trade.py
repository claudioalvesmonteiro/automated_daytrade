'''
TRADE
'''

# 
msft = yf.Ticker("^BVSP")
print(msft)
msft.info

# pegar dados historicos
data = msft.history(period="max")

# reset index
#data.reset_index(inplace=True)

# select columns
data = data[['Close']]

# transform in datetime
data.index = pd.to_datetime(data.index)

# visu
import plotly.plotly as ply
import cufflinks as cf

data.iplot(title="IBOV")

#==================================
# dale
#==================================

# data

def modellingSARIMA(data, target, pred_intervals):

    # importar modelo
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # executar modelo
    model = SARIMAX(data[target], order=(1, 0, 0), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # previsoes para proximos pred_intervals 
    predicted = model_fit.predict(len(data), (len(data)+pred_intervals) )   

    # retornar valores preditos
    return predicted 