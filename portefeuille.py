#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 23:27:55 2020

@author: rouyrrerodolphe
"""


import courbes2

codesISIN = ['AN8068571086', 'BE0003470755', 'BE0974310428', 'CH0012214059',
             'FI0009000681', 'FR0000031122', 'FR0000031577', 'FR0000031684',
             'FR0000031775', 'FR0000032658', 'IBM-1']

for i in range(len(codesISIN)) :
    #nomFic = '{0}.csv' .format(codesISIN[i])
    df = courbes2.cours(codesISIN[i])
    df = courbes2.ATR(df)
    df = courbes2.MM(df,codesISIN[i])
    df = courbes2.MACD(df)
    df = courbes2.RSI(df, 25)
    #plt.plot(df['date'], df['RSI'])
    #bollinger(df)
    
    
    
    