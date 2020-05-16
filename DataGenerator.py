# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os 
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# %%
# data_file = pd.read_csv('./data/load_data.csv')
# data_file = data_file[['CUSTOMER_ID', 'READING_DATETIME', 'GENERAL_SUPPLY_KWH' ]]
# customers_file = pd.read_csv('./data/household_data.csv')


# %%
# customers = customers_file[customers_file.HAS_GAS_HOT_WATER == 'Y']
# customers_ids = customers_file.CUSTOMER_KEY


# %%
# refined_data = pd.DataFrame()


# %%

# for id in customers_ids:
#     refined_data.append(data_file[data_file.CUSTOMER_ID == id])


# %%
class DataGenerator():
    def __init__(self, data_file, window, date_col, reading_col, split = [0.7, 0.2, 0.1], date_format = "%Y-%m-%d %H:%M:%S"):
        self.window = window + 1
        self.split = split
        self.date_col = date_col
        self.reading_col = reading_col
        self.date_format = date_format
        self.__customer_data = {}
        
        for id in data_file.CUSTOMER_ID.unique():
            self.__split_customer_data(data_file, id)

        self.__process_data()
        print('{0} customers'.format(len(self.__customer_data)))
    
    def __split_customer_data(self,data, customer_id):
        self.__customer_data[customer_id] = data[data.CUSTOMER_ID == customer_id]

    def __process_data(self): 
        for customers_id, data in self.__customer_data.items():
            E = []
            D = []
            I = []
            H = []

            for index in range(len(data)):
                E.append(data.iloc[index, self.reading_col])

                date_time = data.iloc[index, self.date_col]
                try:
                    date_time = datetime.strptime(date_time, self.date_format)
                except:
                    date_time = datetime.strptime(date_time, "%Y/%m/%d %H:%M:%S")
        
                D.append(date_time.weekday())
                I.append((date_time.hour * 2) + int(date_time.minute / 30))
    
                if(date_time.weekday() > 4):
                    H.append(1)
                else:
                    H.append(0)

            D = np.asarray(D)
            I = np.asarray(I)
            H = np.asarray(H)
            E = np.asarray(E)

            enc = OneHotEncoder(handle_unknown='ignore')

            enc.fit(D. reshape(-1,1))
            D = enc.transform(D.reshape(-1, 1)).toarray()
            enc.fit(I.reshape(-1,1))
            I = enc.transform(I.reshape(-1, 1)).toarray()
            enc.fit(H.reshape(-1,1))
            H = enc.transform(H.reshape(-1, 1)).toarray()

            E = E.reshape(-1,1)
            #print(E.shape, I.shape, D.shape, H.shape)
            data = np.concatenate((E, I, D, H), axis=1)
            
            sequences = []
            for index in range(len(data) - self.window):
                sequences.append(data[index: index + self.window])

            sequences = np.asarray(sequences)
            sequences = sequences[:, :]
            x_data = sequences[:, :-1]
            y_data = sequences[:, -1][:, 0]

            total = len(x_data)
            train_start = 0
            val_start = int(total * self.split[0])
            test_start = int(total * self.split[0]) + int(total * self.split[1])

            splits = {'train_x': x_data[train_start : val_start], 'train_y': y_data[train_start : val_start],
                      'val_x': x_data[val_start : test_start], 'val_y': y_data[val_start : test_start],
                      'test_x': x_data[test_start : ], 'test_y': y_data[test_start : ]}

            self.__customer_data[customers_id] = splits

    def generate_data(self):
        for id ,data in self.__customer_data.items():
            yield [id, data]


# %%
# dataset= Dataset(data_file, 6, 0, 1, [0.9,0.0,0.1], "%d/%m%Y %H:%M")


# %%
# for data in dataset.get_customer_data():
#     for key,vals in data[1].items():
#         print(key, vals.shape)
#         if(key == 'train_x' or key == 'train_y'):
#             print(vals[:1])

