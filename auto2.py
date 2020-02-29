'''
AUTO FORECAST DAYTRADE
'''

trade = '^BVSP'

# funcao para importar dados
def getTrade(trade, periods):
    import yfinance as yf
    import pandas as pd
    # import data 
    msft = yf.Ticker(trade)
    data = msft.history(period="max")
    # select columns
    data = data[['Close']]
    # transform in datetime
    data.index = pd.to_datetime(data.index)

    return data[len(data)-periods:len(data)]


def generatePlotData(model, data, interval, future_forecast):
    import pandas as pd

    if model == 'autoarima':
        prediction = pd.DataFrame(interval, future_forecast)
        prediction.reset_index(inplace=True)
        print(prediction.head())
        prediction.columns = ['Close', 'Date']
        prediction['Date'] = prediction['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        data['Date'] = data['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        datapred = pd.merge(data, prediction, how = 'outer', on = 'Date')
    elif model == 'prophet':
        prediction = future_forecast[['yhat', 'ds']]
        prediction['ds'] = prediction['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
        data['ds'] = data['ds'].apply(lambda x: x.strftime('%Y-%m-%d'))
        datapred = pd.merge(data, prediction, how = 'outer', on = 'ds')

    # rename target columns and merge outer
    datapred.columns = ['date', 'valor original', 'valor predito']
    return datapred


def autoModels(model, data, start_date, end_date):

    import pandas as pd

    # define interval
    interval = pd.date_range(start=start_date, end=end_date, freq='d') 
    pred_intervals = (len(interval)-1)

    # index to date
    data.reset_index(inplace=True)

    # AUTOARIMA
    if model == 'autoarima':
        from pyramid.arima import auto_arima

        print('AutoARIMA generation: ')

        stepwise_model = auto_arima(data['Close'][0:800], 
                                    trace=True, 
                                    error_action='ignore', 
                                    suppress_warnings=True,
                                    start_p=1, start_q=1,
                                    max_p=3, max_q=3, m=12,
                                    start_P=0, seasonal=True,
                                    d=1, D=1)
        print(stepwise_model.aic())

        stepwise_model.fit(data[['Close']][0:800])
        future_forecast = stepwise_model.predict(n_periods=pred_intervals+1)
    
    # facebook PROPHET
    elif model == 'prophet':
        from fbprophet import Prophet

        print('PROPHET generation: ')

        pdf = data
        pdf.columns = ['ds', 'y']

        prophet_model = Prophet()
        prophet_model.fit(pdf)
        future = prophet_model.make_future_dataframe(periods=pred_intervals)
        future_forecast = prophet_model.predict(future)

    # capture data
    datapred = generatePlotData(model, data, interval, future_forecast)
    
    return datapred

def layLine(min, max):
    import plotly.graph_objs as go
    layout = go.Layout( {'legend': {'bgcolor': '#F5F6F9', 'font': {'color': '#4D5663'}},
                   'paper_bgcolor': '#F5F6F9',
                   'plot_bgcolor': '#F5F6F9',
                   'title': {'font': {'color': '#4D5663'}},
                   'xaxis': {'gridcolor': '#E1E5ED',
                             'showgrid': True,
                             'tickfont': {'color': '#4D5663'},
                             'title': {'font': {'color': '#4D5663'}, 'text': ''},
                             'zerolinecolor': '#E1E5ED'},
                   'yaxis': {'gridcolor': '#E1E5ED',
                             'range' : (min, max),
                             'showgrid': True,
                             'tickfont': {'color': '#4D5663'},
                             'title': {'font': {'color': '#4D5663'}, 'text': ''},
                             'zerolinecolor': '#E1E5ED'}}
    )
    return layout


def predPlot(datapred):

    import plotly.plotly as ply
    import cufflinks as cf

    # min and max on plot
    mini = datapred['valor original'].min()-datapred['valor original'].min()*0.15
    maxi = datapred['valor predito'].max()+datapred['valor predito'].max()*0.15
    # definir layout
    layout = layLine(mini, maxi)

    # index to layout
    datapred.set_index('date', inplace=True)
    
    # forecast visualization
    datapred[['valor original', 'valor predito']].iplot(mode='lines',
                                    size = 8,
                                    colors=['#87cded', 'pink'],
                                    layout=layout.to_plotly_json(),
                                    filename='test')


def gerar_previsao(indice, modelo, periodos_anteriores, start_date, end_date):
    # get trade
    data = getTrade(indice, periodos_anteriores)
    # get prediction
    predicted = autoModels(modelo, data, start_date, end_date)
    # plot prediction
    predPlot(predicted)