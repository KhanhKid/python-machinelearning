from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt, expm1
import numpy as np
from sklearn.metrics import mean_squared_error
from numpy import genfromtxt, savetxt

#Số Lượng Quận
arrQuan = ['q1','q2','q3','q4','q5','q6','q7','q8','q9','q10','q11','q12','binh_chanh','binh_tan','binh_thanh','can_gio','cu_chi','go_vap','hooc_mon','nha_be','phu_nhuan','tan_binh','tan_phu','thu_duc']
arrDayInMonth = {4:30, 3:31}

for district in range(0,24): 
    #Dự báo tháng 3 + 4
    for month in range(3,5): 
        print("Tháng:%d -> Quận:%s" % (month ,arrQuan[district]))
        # read database month from csv 
        series = read_csv('tempature_2019/'+str(month)+'.csv', header=-1, parse_dates=[0], index_col=0, squeeze=True)
        X = series.values
        train = X[0:len(X),district]
        dataset = train
        predictions = list()
        dataset = [x for x in train]
        numYear = 4 # số năm trong quá khứ
        numDate = arrDayInMonth[month]
        for t in range(numDate):
            history = [];
            # Đọc dữ liệu quá khứ và import vào history
            for year in range(0,numYear):
                history.append(dataset[t+(year*numDate)])
            model = ARIMA(history, order=(1,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat[0])
            history.append(yhat[0])
            print('Ngày %d -> predicted=%f' % (t+1 ,yhat[0]))

        savetxt("Result/"+str(district)+"_"+arrQuan[district]+"_"+str(month)+".txt", predictions, delimiter=',')
        pass