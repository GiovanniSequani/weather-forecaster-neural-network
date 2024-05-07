import pandas as pd
import numpy as np
import datetime

data = pd.read_csv("datasets\\padova.csv")

# sezione colonne
columns = ["date_time","precipMM","sunrise","sunset","DewPointC","FeelsLikeC","HeatIndexC","WindGustKmph","cloudcover","humidity","pressure","tempC","winddirDegree","windspeedKmph"]
data = data[columns]

# sunminutes
sunrise = data["sunrise"].apply(lambda x: datetime.datetime.strptime(x, "%I:%M %p"))
sunset = data["sunset"].apply(lambda x: datetime.datetime.strptime(x, "%I:%M %p"))
data["sunmin"] = (sunset - sunrise).apply(lambda x: x.total_seconds()/60)
data.drop(["sunrise","sunset"], axis=1, inplace=True)

# lag:
def crea_lagged(data, lags=24, no_lag=[]):
    def lags_values(values, lags, name):
        df = pd.DataFrame({name:values})
        for lag in range(1, lags+1):
            col_name = f"{name}_{lag}"
            df[col_name] = df[name].shift(lag)
        return df

    df = pd.DataFrame()
    for col in data.columns:
        if not col in no_lag:
            df = pd.concat([df, lags_values(data[col], lags, col)], axis=1)
        else:
            df = pd.concat([df, data[col]], axis=1)
    return df

data_lagged = crea_lagged(data, no_lag = ["date_time"])

# variabile target
SHIFT = 2
data_lagged["precipBin"] = np.where(data_lagged["precipMM"] == 0.0, 0, 1)
data_lagged["precipBin"] = data_lagged["precipBin"].shift(-SHIFT)

# rimuovo NaN values e salvo
data_lagged.dropna(inplace=True)
data_lagged.to_csv("datasets\\padova_lag.csv", index=False)
