#import librarias

import streamlit as st
import pickle
import pandas as pd
from PIL import Image


#extraer archivos pickle

with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)


with open('log_reg1.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)


with open('svc_m.pkl', 'rb') as sv:
    svc_m = pickle.load(sv)

#funcion para clasificar
def classify(num):
    if num == 0:
        return "No hacer descuento"
    else:
        return "Hacer descuento"

image = Image.open('/Users/RicardoVReig/Desktop/TFM_APP_tienda_modelo_tabular/streamlit/logo_mercadona.png')





def main():
    st.image(image,caption=None, width=100, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    st.title('Predicción de venta diaria frescos Mercadona para aplicar descuento')

    st.sidebar.header('Parámetros imputables')
#poner parametros para elegir:
    def user_input_parameters():
        hora1 = st.sidebar.slider('Ventas 8-9 AM', 0, 500, 20)
        hora2 = st.sidebar.slider('Ventas 9-10 AM', 0, 500, 20)
        hora3 = st.sidebar.slider('Ventas 10-11 AM', 0, 500, 20)
        hora4 = st.sidebar.slider('Ventas 11-12 AM', 0, 500, 20)
        hora5 = st.sidebar.slider('Ventas 12-13 AM', 0, 500, 20)
        hora6 = st.sidebar.slider('Ventas 13-14 AM', 0, 500, 20)
        hora7 = st.sidebar.slider('Ventas 14-15 AM', 0, 500, 20)
        hora8 = st.sidebar.slider('Ventas 15-16 AM', 0, 500, 20)
        hora9 = st.sidebar.slider('Ventas 16-17 AM', 0, 500, 20)
        hora10 = st.sidebar.slider('Ventas 17-18 AM', 0, 500, 20)
        hora11= st.sidebar.slider('Ventas 18-19 AM', 0, 500, 20)
        hora12= st.sidebar.slider('Ventas 19-20 AM', 0, 500, 20)
        hora13= st.sidebar.slider('Ventas 20-21 AM', 0, 500, 20)
        STO_PESO_REAL = st.sidebar.slider('Stock Inicial', 10, 5000, 300)
        festivo = st.sidebar.slider('Es festivo? 0 = No, 1 = Sí!', 0, 1, 0)
        dsemana_J = st.sidebar.slider('Es jueves? 0 = No, 1 = Si', 0, 1, 0)
        dsemana_L = st.sidebar.slider('Es lunes? 0 = No, 1 = Si :(', 0, 1, 0)
        dsemana_M = st.sidebar.slider('Es martes? 0 = No, 1 = Si', 0, 1, 0)
        dsemana_S = st.sidebar.slider('Es sábado? 0 = No, 1 = Si :)', 0, 1, 0)
        dsemana_V = st.sidebar.slider('Es viernes? 0 = No, 1 = Si', 0, 1, 0)
        dsemana_X = st.sidebar.slider('Es miercoles? 0 = No, 1 = Si', 0, 1, 0)
        data = {'hora1':hora1,
                'hora2':hora2,
                'hora3':hora3,
                'hora4':hora4,
                'hora5':hora5,
                'hora6':hora6,
                'hora7':hora7,
                'hora8':hora8,
                'hora9':hora9,
                'hora10':hora10,
                'hora11':hora11,
                'hora12':hora12,
                'hora13':hora13,
                'STO_PESO_REAL': STO_PESO_REAL,
                'festivo':festivo,
                'dsemana_J':dsemana_J,
                'dsemana_L':dsemana_L,
                'dsemana_M':dsemana_M,
                'dsemana_S':dsemana_S,
                'dsemana_V':dsemana_V,
                'dsemana_X':dsemana_X,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

#escoger el modelo que queramos probar:
    option = ['Linear Regression', 'Logistic Regression', 'SVM']
    model = st.sidebar.selectbox('Qué modelo quieres elegir?', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Linear Regression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classify(log_reg.predict(df)))
        else: 
            st.success(classify(svc_m.predict(df)))


if __name__ == "__main__":
    main()
