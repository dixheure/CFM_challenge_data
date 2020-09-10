# -*- coding: utf-8 -*-
"""
Created on Sat Apr  18 12:24:17 2020

"""

import pandas as pd

class CFMutil:
    
    def __init__(self):
        pass
    
    def get_orderbook_trade_filtred(self, index_data, train_data):
        
        extracted_train_data = []
        # look for the needed columns name in the dataframe, example '(0'
        for i in train_data.columns:    
            if '('+index_data in str(i):
                extracted_train_data.append(i)
        
        #extract from the dataframe needed column
        extracted_train_data = train_data.loc[:, [v for v in extracted_train_data]]
        
        trade = []
        # look for the needed columns name (trade) in the dataframe
        for i in extracted_train_data.columns:    
            if 'qty' in str(i):
                trade.append(i)
            if 'tod' in str(i):
                trade.append(i)
                #sort_trade = i
            if 'price' in str(i):
                trade.append(i)
            if 'source_id' in str(i):
                trade.append(i)
        #extract from the dataframe needed column on trade
        trade = extracted_train_data.loc[:, [v for v in trade]]
        #drop trade from dataframe so that we get orderbook
        orderbook = extracted_train_data.drop([v for v in trade.columns], axis=1)

        #trade = trade.sort_values(by=[sort_trade])
        #orderbook = orderbook.sort_values(by=[sort_orderbook])

        return orderbook, trade
    
    def get_sorted_orderbook_trade(self, train_data, train_mode=True):
        
        orderbook_0, trade_0 = CFMutil.get_orderbook_trade_filtred(None, '0', train_data)
        orderbook_1, trade_1 = CFMutil.get_orderbook_trade_filtred(None, '1', train_data)
        orderbook_2, trade_2 = CFMutil.get_orderbook_trade_filtred(None, '2', train_data)
        orderbook_3, trade_3 = CFMutil.get_orderbook_trade_filtred(None, '3', train_data)
        orderbook_4, trade_4 = CFMutil.get_orderbook_trade_filtred(None, '4', train_data)
        orderbook_5, trade_5 = CFMutil.get_orderbook_trade_filtred(None, '5', train_data)
        orderbook_6, trade_6 = CFMutil.get_orderbook_trade_filtred(None, '6', train_data)
        orderbook_7, trade_7 = CFMutil.get_orderbook_trade_filtred(None, '7', train_data)
        orderbook_8, trade_8 = CFMutil.get_orderbook_trade_filtred(None, '8', train_data)
        orderbook_9, trade_9 = CFMutil.get_orderbook_trade_filtred(None, '9', train_data)        
        
        trade = pd.concat([trade_0, trade_1, trade_2, trade_3, trade_4, trade_5, trade_6, trade_7, trade_8, trade_9], axis=1)
        orderbook = pd.concat([orderbook_0, orderbook_1, orderbook_2, orderbook_3, orderbook_4, orderbook_5, orderbook_6, orderbook_7, orderbook_8, orderbook_9], axis=1)
        
        if (train_mode==True):
            #we add commun column to trade and orderbook
            trade = pd.concat([trade, train_data[['stock_id', 'day_id', 'ID', 'source_id']]], axis=1)
            orderbook = pd.concat([orderbook, train_data[['stock_id', 'day_id', 'ID', 'source_id']]], axis=1)
        else:
            #we add commun column to trade and orderbook
            trade = pd.concat([trade, train_data[['stock_id', 'day_id', 'ID']]], axis=1)
            orderbook = pd.concat([orderbook, train_data[['stock_id', 'day_id', 'ID']]], axis=1)
        
        #trade.sort_values(by=['day_id', 'tod'])
        #orderbook.sort_values(by=['day_id', 'ts_last_update'])
        
        sort_trade = []
        for i in trade.columns:
            if 'tod' in str(i):
                sort_trade.append(i)
         
        sort_trade.insert(0, 'day_id')
               
        trade.sort_values(by=[v for v in sort_trade], ascending=False, inplace=True)
        
        sort_orderbook = []
        for i in orderbook.columns:
            if 'last_update' in str(i):
                sort_orderbook.append(i)
         
        sort_orderbook.insert(0, 'day_id')
               
        orderbook.sort_values(by=[v for v in sort_orderbook], ascending=False, inplace=True)
        
        
        return orderbook, trade
    
    def get_sorted_labels(self, labels_data, train_trade):
        # sort output based on ID from train_trade 
        labels_data = labels_data.reindex(train_trade['ID'])
        
        return labels_data
        
    def get_splited_orderbook(train_orderbook, idx):
        if (idx==0):
            splited_train_orderbook = train_orderbook.iloc[:, 0:9]
        else:
            start = idx*9+1
            end = idx*9+10
            splited_train_orderbook = train_orderbook.iloc[:, start:end]
        
        splited_train_orderbook = pd.concat([splited_train_orderbook, train_orderbook[['stock_id', 'day_id', 'ID']]], axis=1)
        
        return splited_train_orderbook
 
    