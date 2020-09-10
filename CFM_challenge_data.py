# -*- coding: utf-8 -*-
"""
Created on Sat Apr  18 10:39:06 2020

"""


import pandas as pd
import numpy as np

from keras.utils import to_categorical

from CFM_util import CFMutil
from keras.models import load_model

#LSTM config
timesteps_orderbook = 6
timesteps_trade = 10
subsequences = 2
subsequences_prediction = 1
LSTM_units = 50
dropout = 0.1
epochs = 100
traning_batch_size = 100
prediction_batch_size = 1


############################################################################## 
############################################################################## DATA PREPARATION
##############################################################################

# Importing training set
row_train_data = pd.read_hdf("train_dc2020.h5", 'data')
row_labels_data = pd.read_csv("train_labels.csv")


# look for where are index and coloumn row nan values
idx, idy = np.where(pd.isnull(row_train_data))
# remove duplicated value in index row
idx = np.unique(idx)
# remove indexes where there at least one nan
row_train_data.drop(index=idx.tolist(), axis=0, inplace=True)
row_labels_data.drop(index=idx.tolist(), axis=0, inplace=True)    

train_data = row_train_data
labels_data = row_labels_data

train_data=pd.concat([train_data, labels_data['source_id']], axis=1)

#extract orderbook and trade from train_data
train_orderbook, train_trade = CFMutil.get_sorted_orderbook_trade(None, train_data, True)
train_labels = CFMutil.get_sorted_labels(None, labels_data, train_trade)

######## Reshaping Orderbooks

df = train_orderbook
a = df.iloc[:, :-4].to_numpy()
b = df.iloc[:, -4:].to_numpy()

c = a.reshape(-1, 9)
k = c.shape[0] / a.shape[0]
d = b.repeat(k, axis=0)

reshaped_train_orderbook = pd.DataFrame(np.column_stack([c, d]), columns=['ask', 'ask1', 'ask_size', 'ask_size1', 'bid', 'bid1', 'bid_size','bid_size1', 'ts_last_update', 'stock_id', 'day_id', 'ID', 'source_id'])

reshaped_train_orderbook = reshaped_train_orderbook.sort_values(['source_id', 'day_id', 'ts_last_update'], ascending=False)

y_train_orderbook = reshaped_train_orderbook['source_id'].values
y_train_orderbook = y_train_orderbook.reshape((y_train_orderbook.shape[0]), 1)

x_train_orderbook = reshaped_train_orderbook
x_train_orderbook = x_train_orderbook.drop(['ts_last_update', 'day_id', 'ID', 'source_id'], axis=1)
x_train_orderbook = x_train_orderbook.values
x_train_orderbook = x_train_orderbook.reshape(x_train_orderbook.shape[0], x_train_orderbook.shape[1])

# shift the target sample by one step, otherwise we get trouble with generator - TimeseriesGenerator
y_train_orderbook = np.insert(y_train_orderbook, 0, 0)
y_train_orderbook = np.delete(y_train_orderbook, -1)
y_train_orderbook = to_categorical(y_train_orderbook)

######## Reshaping Trades 

df = train_trade
a = df.iloc[:, :-4].to_numpy()
b = df.iloc[:, -4:].to_numpy()

c = a.reshape(-1, 4)
k = c.shape[0] / a.shape[0]
d = b.repeat(k, axis=0)

reshaped_train_trade = pd.DataFrame(np.column_stack([c, d]), columns=['price', 'qty', 'trade_source_id', 'tod', 'stock_id', 'day_id', 'ID', 'source_id'])

reshaped_train_trade = reshaped_train_trade.sort_values(['trade_source_id', 'day_id', 'tod'], ascending=False)

y_train_trade = reshaped_train_trade['source_id'].values
y_train_trade = y_train_trade.reshape((y_train_trade.shape[0]), 1)

x_train_trade = reshaped_train_trade
x_train_trade = x_train_trade.drop(['tod', 'day_id', 'ID', 'source_id'], axis=1)
x_train_trade = x_train_trade.values
x_train_trade = x_train_trade.reshape(x_train_trade.shape[0], x_train_trade.shape[1])

# shift the target sample by one step, otherwise we get trouble with generator - TimeseriesGenerator
y_train_trade = np.insert(y_train_trade, 0, 0)
y_train_trade = np.delete(y_train_trade, -1)
y_train_trade = to_categorical(y_train_trade)


############################################################################## 
############################################################################## MULTI-HEADED MODELELING
##############################################################################

from keras.utils import Sequence
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate

from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten

######## Creating a generator to train the modele 
class MultipleHeadCNNLSTM_Generator(Sequence):
    def __init__(self, x_train_orderbook, y_train_orderbook, x_train_trade, y_train_trade, features, timesteps_orderbook, timesteps_trade, subsequences, batch_size, istrain):
        self.gen_orderbook = TimeseriesGenerator(x_train_orderbook, y_train_orderbook, length=timesteps_orderbook, batch_size=batch_size, stride=timesteps_orderbook)
        self.gen_trade = TimeseriesGenerator(x_train_trade, y_train_trade, length=timesteps_trade, batch_size=batch_size, stride=timesteps_trade)

        self.features = features
        self.subsequences = subsequences
        self.timesteps_orderbook = timesteps_orderbook
        self.timesteps_trade = timesteps_trade
        self.batch_size = batch_size
        self.istrain = istrain

    def __len__(self):
        #gen_orderbook and gen_trade have the same len
        return self.gen_orderbook.__len__()-1        

    def __getitem__(self, idx):
        X1_batch, Y1_batch = self.gen_orderbook.__getitem__(idx)
        X2_batch, Y2_batch = self.gen_trade.__getitem__(idx)
      
        X1_batch = np.reshape(X1_batch, (-1, self.subsequences, self.timesteps_orderbook, self.features[0]))
        X2_batch = np.reshape(X2_batch, (-1, self.subsequences, self.timesteps_trade, self.features[1]))
        
        if (self.istrain == True):
            Y1_batch = np.reshape(Y1_batch, (-1, 6))
            Y1_batch = Y1_batch[0:int(self.batch_size/self.subsequences), :]
            Y2_batch = np.reshape(Y2_batch, (-1, 6))
            Y2_batch = Y2_batch[0:int(self.batch_size/self.subsequences), :]
                   
        X_batch = [X1_batch, X2_batch]
        Y_batch = Y1_batch
        
        #return X_batch, Y_batch #for multi-headed model
        return X_batch, Y_batch
    
def multi_headed_model(features, LSTM_units, dropout, timesteps_orderbook, timesteps_trade, batch_size):
        
        # Orderbook Head
        input_orderbook = Input(shape=(None, timesteps_orderbook, features[0]))
        con1d_orderbook = (TimeDistributed(Conv1D(filters=features[0], kernel_size=1, activation='relu')))(input_orderbook)
        dropout_orderbook = (Dropout(dropout))(con1d_orderbook)
        maxpooling_orderbook = (TimeDistributed(MaxPooling1D(pool_size=2)))(dropout_orderbook)
        flatten_orderbook = (TimeDistributed(Flatten()))(maxpooling_orderbook)

        batchNormalization_orderbook = BatchNormalization()(flatten_orderbook)
        lstm1_orderbook = LSTM(units = LSTM_units, return_sequences = True)(batchNormalization_orderbook)
        dropout1_orderbook = Dropout(dropout)(lstm1_orderbook)
        lstm2_orderbook = LSTM(units = LSTM_units)(dropout1_orderbook)
        dropout2_orderbook = Dropout(dropout)(lstm2_orderbook)
        
        # Trade Head
        input_trade = Input(shape=(None, timesteps_trade, features[1]))
        con1d_trade = (TimeDistributed(Conv1D(filters=features[1], kernel_size=1, activation='relu')))(input_trade)
        dropout_trade = (Dropout(dropout))(con1d_trade)
        maxpooling_trade = (TimeDistributed(MaxPooling1D(pool_size=2)))(dropout_trade)
        flatten_trade = (TimeDistributed(Flatten()))(maxpooling_trade)
        
        batchNormalization_trade = BatchNormalization()(flatten_trade)
        lstm1_trade = LSTM(units = LSTM_units, return_sequences = True)(batchNormalization_trade)
        dropout1_trade = Dropout(dropout)(lstm1_trade)
        lstm2_trade = LSTM(units = LSTM_units)(dropout1_trade)
        dropout2_trade = Dropout(dropout)(lstm2_trade)
    
        # Merge
        merged = concatenate([dropout2_orderbook, dropout2_trade])
        
        # Interpretation
        outputs = Dense(6, activation='softmax')(merged)
        model = Model(inputs=[input_orderbook, input_trade], outputs=outputs)

        return model
        
############################################################################## 
############################################################################## MULTI-HEADED MODELE TRAINING
##############################################################################
        
features = [x_train_orderbook.shape[1], x_train_trade.shape[1]]
generator = MultipleHeadCNNLSTM_Generator(x_train_orderbook, y_train_orderbook, x_train_trade, y_train_trade, features, timesteps_orderbook, timesteps_trade, subsequences, traning_batch_size, True)

train_multi_headed_model = multi_headed_model(features, LSTM_units, dropout, timesteps_orderbook, timesteps_trade, traning_batch_size)

train_multi_headed_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


train_multi_headed_model.fit_generator(generator, epochs=epochs, steps_per_epoch=traning_batch_size, workers=2, verbose=2)

train_multi_headed_model.save('CFM_Challenge_data.h5')

############################################################################## 
############################################################################## PREDICTION
##############################################################################

train_multi_headed_model = load_model('CFM_Challenge_data.h5')

row_test_data = pd.read_hdf("test_dc2020.h5", 'data')

# propagate non-null values forward 
row_test_data.fillna(method='ffill', inplace=True)
row_test_data = row_test_data.sort_values(['day_id'], ascending=False)

test_data = row_test_data

test_orderbook, test_trade = CFMutil.get_sorted_orderbook_trade(None, test_data, False)


######################################### RESHAPING ORDERBOOKS


df = test_orderbook
a = df.iloc[:, :-3].to_numpy()
b = df.iloc[:, -3:].to_numpy()

c = a.reshape(-1, 9)
k = c.shape[0] / a.shape[0]
d = b.repeat(k, axis=0)

reshaped_test_orderbook = pd.DataFrame(np.column_stack([c, d]), columns=['ask', 'ask1', 'ask_size', 'ask_size1', 'bid', 'bid1', 'bid_size','bid_size1', 'ts_last_update', 'stock_id', 'day_id', 'ID'])

reshaped_test_orderbook = reshaped_test_orderbook.sort_values(['day_id', 'ts_last_update'], ascending=False)

x_test_orderbook = reshaped_test_orderbook
x_test_orderbook = x_test_orderbook.drop(['ts_last_update', 'day_id', 'ID'], axis=1)
x_test_orderbook = x_test_orderbook.values
x_test_orderbook = x_test_orderbook.reshape(x_test_orderbook.shape[0], x_test_orderbook.shape[1])

y_test_orderbook = np.zeros((x_test_orderbook.shape[0], ))
y_test_orderbook = to_categorical(y_test_orderbook)


######################################### RESHAPING TRADES

df = test_trade
a = df.iloc[:, :-3].to_numpy()
b = df.iloc[:, -3:].to_numpy()

c = a.reshape(-1, 4)
k = c.shape[0] / a.shape[0]
d = b.repeat(k, axis=0)

reshaped_test_trade = pd.DataFrame(np.column_stack([c, d]), columns=['price', 'qty', 'trade_source_id', 'tod', 'stock_id', 'day_id', 'ID'])

reshaped_test_trade = reshaped_test_trade.sort_values(['trade_source_id', 'day_id', 'tod'], ascending=False)

x_test_trade = reshaped_test_trade
x_test_trade = x_test_trade.drop(['tod', 'day_id', 'ID'], axis=1)
x_test_trade = x_test_trade.values
x_test_trade = x_test_trade.reshape(x_test_trade.shape[0], x_test_trade.shape[1])

y_test_trade = np.zeros((x_test_trade.shape[0], ))
y_test_trade = to_categorical(y_test_trade)



######################################### MULTI-HEADED MODELE PREDICTION

trained_weights = train_multi_headed_model.get_weights()

features = [x_test_orderbook.shape[1], x_test_trade.shape[1]]

test_generator = MultipleHeadCNNLSTM_Generator(x_test_orderbook, y_test_orderbook, x_test_trade, y_test_trade, features, timesteps_orderbook, timesteps_trade, subsequences_prediction, prediction_batch_size, False)

prediction_multi_headed_model = multi_headed_model(features, test_generator, LSTM_units, dropout, timesteps_orderbook, timesteps_trade, prediction_batch_size)

prediction_multi_headed_model.set_weights(trained_weights)
# compile model
prediction_multi_headed_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

predicted_source_id = prediction_multi_headed_model.predict_generator(generator=test_generator, workers=2, verbose=1)

result_source_id = np.argmax(predicted_source_id, axis=1) 


######################################### PREPARING RESULT TO BE SUBMITTED

result_source_id = pd.DataFrame(data=result_source_id, columns=['source_id'])

indexes = np.unique(reshaped_test_trade['ID'].values, return_index=True)[1]
get_unique_ID = [reshaped_test_trade['ID'].values[index] for index in sorted(indexes)]


get_unique_ID = pd.DataFrame(data=get_unique_ID, columns=['ID'])

cfm_result = pd.concat([get_unique_ID, result_source_id], ignore_index=True, axis=1, sort=False)
cfm_result.columns = ['ID', 'source_id']
cfm_result['ID'] = cfm_result['ID'].astype(np.int64)

cfm_result.to_csv('baseline.csv', index=False)


