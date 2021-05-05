
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import performance_metrics
from plotly import graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error

#Set de la page d'application et de l'onglet browser
st.set_page_config( page_title="Application de prédiction des prix", page_icon="W", layout="wide", initial_sidebar_state="expanded")

#variable de départ et aujourd'hui
START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#Titre de la page
st.title('Application de prédiction des prix ')

#création d'une sidebar et text input du symbole yahoo finance à analyser
stocks= st.sidebar.text_input('Quelle action souhaites-tu analyser? (ajoute le symbole seulement!)','GOOG')
st.sidebar.write( 'Action choisie', stocks)


#variable de nombre de jour de prédiction
n_jours = st.sidebar.slider('Quelle plage de prédiction (en jours):', 1, 365)


#chargement de l'historique depuis Yahoo finance
@st.cache
def load_data(ticker):
    data_new = yf.download(ticker, START, TODAY)
    data_new.reset_index(inplace=True)
    return data_new

	
data_load_state = st.text('Chargement des données...')
data_new = load_data(stocks)
data_load_state.text('Données chargées!')
data = data_new.copy()

st.subheader('tableau des données')
st.write(data.tail(50),  use_container_width=True)

   


#création des EMA 
def signals_by_EMA(data):
  
  short_window = 2
  middle_window = 10
  long_window = 60
  data['flag'] = 0.0

  #SMA
  #signals['short_SMA'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
  #signals['middle_SMA'] = data['Close'].rolling(window=middle_window, min_periods=1, center=False).mean()
  #signals['long_SMA'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

  #data['short_SMA'] = signals['short_SMA']
  #data['middle_SMA'] = signals['middle_SMA']
  #data['long_SMA'] = signals['long_SMA']
  
  
  #EMA
  data['short_EMA'] = (data['Close'].ewm(span=short_window,adjust=True,ignore_na=True).mean())
  data['middle_EMA'] = (data['Close'].ewm(span=middle_window,adjust=True,ignore_na=True).mean())
  data['long_EMA'] = (data['Close'].ewm(span=long_window,adjust=True,ignore_na=True).mean())
  
  #data['flag'][short_window:] = np.where(data['short_EMA'][short_window:] > data['middle_EMA'][short_window:], 1.0, 0.0)   
  #data['positions'] = data['flag'].diff()


signals_by_EMA(data)

#Signal d'achat et Vente : methode turtle
def buy_stock(data):
    signal_Buy = []
    signal_Sell = []

    count = int(np.ceil(len(data) * 0.1))
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['trend'] = data['Close']
    signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
    signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
    signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1 #sell 
    signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1 #buy

    data['signal']= signals['signal']

    for i in range (len(data)):
      if data['signal'][i] == 1:
        signal_Buy.append(data['Close'][i])
        signal_Sell.append(np.nan)
      elif data['signal'][i] == -1 :
        signal_Sell.append(data['Close'][i])
        signal_Buy.append(np.nan)
      else:
       signal_Buy.append(np.nan)
       signal_Sell.append(np.nan)

    data['signal_Buy'] = signal_Buy
    data['signal_Sell'] = signal_Sell



buy_stock(data)





# Graphe plotly du cours de l'action et des EMA + signaux d'achat/vente

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines',name="Cours de l'action"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['long_EMA'], mode='lines',name='EMA60'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['middle_EMA'], mode='lines',name="EMA10"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['short_EMA'], mode='lines',name="EMA2"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['signal_Buy'], mode='markers',name="Signal d'achat"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['signal_Sell'], mode='markers',name="Signal de vente"))
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig,  use_container_width=True)

plot_raw_data()


if st.sidebar.button(f'Lancez la prédiction pour {n_jours} jours') :

  with st.spinner("Calcul en cours,  c'est pas long normalement! "):
    # Pre-processing pour fbprophet.
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # prediction faite par Fbprophet
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=n_jours)
    forecast = m.predict(future)


    # afficher le tableau et le graphe de prediction

    st.subheader(f'Prédiction pour {n_jours} jours')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1,  use_container_width=True)

    st.subheader('Tableau de données prédites')
    st.write(forecast.tail(n_jours),use_container_width=True)


    # décomposition de la prédiction ( tendance & saisonnalité)

    st.subheader("Décomposition de la prédiction : tendance / Saisonnalité mensuelle / Saisonalité hebdomadaire")
    fig2 = plot_components_plotly(m, forecast)
    st.plotly_chart(fig2,  use_container_width=False)


    #affichage du nombre de jours prédit dans la sidebar
    st.sidebar.subheader(f"Evaluation de la prédiction sur {n_jours} jours")

    #création dun tableau de comparaison 
    metric_df = forecast.set_index('ds')[['yhat']].join(df_train.set_index('ds').y).reset_index()
    metric_df.dropna(inplace=True)

    #Erreur RMSE/MAE/MAX
    rmse = round(mean_squared_error(metric_df.y, metric_df.yhat, squared = False ), 2)
    mae = round(mean_absolute_error(metric_df.y, metric_df.yhat),2)
    max_error = round(max_error(metric_df.y,metric_df.yhat),2)

    #affichage de l'erreur dans la sidebar.
    st.sidebar.write('Erreur Moyenne - RMSE = ',rmse)
    st.sidebar.write('Erreur Absolue -  MAE = ',mae)
    st.sidebar.write('Erreur Max  -  MaxErr = ',max_error)

    #congratz!!!!! 
    st.balloons()
  st.success("c'est fait! c'était pas long n'est_ce pas? ")

