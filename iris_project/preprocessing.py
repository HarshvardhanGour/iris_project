from sklearn.preprocessing import MinMaxScaler
import pandas as pd
mms=MinMaxScaler(feature_range=(-1,1))

from sklearn.preprocessing import power_transform

def rem_skew(df):
    array_data=power_transform(df,method="yeo-johnson")
    df=pd.DataFrame(array_data,columns=df.columns)
    return df

def scale(df):
    array_data=mms.fit_transform(df)
    df=pd.DataFrame(array_data,columns=df.columns)
    return df
