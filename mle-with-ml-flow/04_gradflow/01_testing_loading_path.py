import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame


start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2020, 12, 31)
btc_df = web.DataReader("BTC-USD", 'yahoo', start, end)
print(btc_df)