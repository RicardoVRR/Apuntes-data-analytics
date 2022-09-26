#importar librerias necesarias
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import pickle
from sklearn.preprocessing import scale


#cargar datos en un dataset
data = pd.read_csv(r"/Users/RicardoVReig/Desktop/TFM_APP_tienda_modelo_tabular/streamlit/finaldf.csv", encoding='utf-8')


#split data
model_df = data.copy()
X = model_df.drop(columns=['overstock'])
y = model_df['overstock']

#tipificar

X_scale = pd.DataFrame(scale(X))
X_scale.columns = X.columns
X = X_scale
X.columns = X_scale.columns
print(X.head())

#separar los datos en entratanmiento y prueba

x_train, x_test, y_train, y_test = train_test_split(X,y)


lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_m = SVC()


#entrenar modelos

lin_regr = lin_reg.fit(x_train, y_train)
log_regr = log_reg.fit(x_train, y_train)
svc_mo = svc_m.fit(x_train, y_train)

#guardar archivos en archivo pickle

with open('lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_regr, li)

#with open('log_reg1.pkl', 'wb') as lo:
#   pickle.dump(log_regr, lo)

with open('svc_m.pkl', 'wb') as sv:
    pickle.dump(svc_mo, sv)