#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:24:49 2020

@author: rouyrrerodolphe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cours(code) : 
    nomFic = '{0}.csv' .format(code)
    
    # Les donnees du fichier sont passees dans un dataframe : 
    df = pd.read_csv(nomFic,sep=';', names=['date', 'ouverture', 'plus haut', 
                                            'plus bas','cloture', 'volume'])
    
    df = df.reset_index()
    df = df.rename(columns={'index' : 'code ISIN'})
    
    # On traite la date pour pouvoir l'utliser au bon format :
    new = df['date'].str.split('/', expand=True)
    new[2] = pd.to_numeric(new[2])
    if new[2][0]>50 : 
        new[2] = new[2] + 1900
    else :
        new[2] = new[2] + 2000
    df['date'] = pd.to_datetime({'year' : new[2], 'month' : new[1], 'day' : new[0]})
    
    # On ordonne le dataframe par ordre chronologique : 
    df = df.sort_values(by='date')
    df = df.reset_index()
    
    return df

def ATR(dataframe):
    
    dataframe['ATR1'] = abs(dataframe['plus haut'] - dataframe['plus bas'])
    dataframe['ATR2'] = abs(dataframe['plus haut'] - dataframe['cloture'].shift())
    dataframe['ATR3'] = abs(dataframe['plus bas'] - dataframe['cloture'].shift())
    dataframe['ATR'] = dataframe[['ATR1', 'ATR2', 'ATR3']].max(axis=1)
    
    dataframe = dataframe.drop(['ATR1','ATR2','ATR3'], axis=1)
    
    return dataframe
    
def MM(df,code) :
    
    # Calcul des 2 moyennes mobiles 20 et 50 :
    df['MMA20'] = df['cloture'].rolling(window=20).mean() 
    df['MMA50'] = df['cloture'].rolling(window=50).mean()
    
    # Intersections des courbes des 2 moyennes mobiles :
    idx = np.argwhere(np.diff(np.sign(df['MMA20'] - df['MMA50'])) != 0).reshape(-1) + 0
    
    fig, ax = plt.subplots()
    plt.plot(df['date'], df['cloture'])
    plt.plot(df['date'], df['MMA20'])
    plt.plot(df['date'], df['MMA50'])
    plt.plot(df['date'][idx], df['MMA50'][idx], 'ro')
    fig.suptitle(code, fontsize=16)
    
    #df['position'][idx] = np.where(df['MMA20'][idx]>df['MMA50'][idx], 'achat', df['position'][idx]) 
    #df['position'][idx] = np.where(df['MMA20'][idx]<df['MMA50'][idx], 'vente', df['position'][idx])
    
    print('nomre de positions MM :', len(idx))
    
    return df
    
def MACD(df) :
    
    # Moyennes mobiles exponentielles :
    df['MME12'] = df['cloture'].ewm(span=12,adjust=False).mean()
    df['MME26'] = df['cloture'].ewm(span=26,adjust=False).mean()
    
    # MACD : 
    df['MACD'] = df['MME12'] - df['MME26']
    df['signal'] = df['MACD'].ewm(span=9,adjust=False).mean()
    # Le signal est la MME9 de la MACD
    
    df['signe'] = np.zeros(len(df))
    df['signe'] = np.where((df['MACD'] - df['signal'])>0, 'achat', df['signe'])
    df['signe'] = np.where((df['MACD'] - df['signal'])<0, 'vente', df['signe'])

    # Intersection de la MACD et du signal : 
    idx2 = np.argwhere(np.diff(np.sign(df['MACD'] - df['signal'])) != 0).reshape(-1) + 0
    
    # Pour savoir quand prendre position : (il faudrait verifier que la position dure (14 jours))
    df['position'] = np.zeros(len(df))
    df['position'][idx2] = np.where(df['MACD'][idx2]>df['signal'][idx2], 'achat', df['position'][idx2]) 
    df['position'][idx2] = np.where(df['MACD'][idx2]<df['signal'][idx2], 'vente', df['position'][idx2])

    print('nombre de positions MACD :', len(idx2))
    
    """
    plt.plot(df['date'], df['MACD'])
    plt.plot(df['date'], df['MME-MACD'])
    plt.plot(df['date'][idx2], df['MACD'][idx2], 'ro')
    """
    
    return df    
    
def bollinger(df) :
    
    # Bandes de Bollinger :
    df['ecart-type'] = df['cloture'].rolling(window=20).std()
    df['BBH'] = df['MMA20'] + 2*df['ecart-type']
    df['BBB'] = df['MMA20'] - 2*df['ecart-type']
    
    df['largeur'] = df['BBH'] - df['BBB']
    df['positions'] = np.where(df['largeur'] < 8, 'achat', 'rien')
    
    # set style, empty figure and axes
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    
    # Get index values for the X axis for facebook DataFrame
    x_axis = df.index.get_level_values(0)
    
    # Plot shaded 21 Day Bollinger Band for Facebook
    ax.fill_between(x_axis, df['BBH'], df['BBB'], color='grey')
    
    # Plot Adjust Closing Price and Moving Averages
    ax.plot(x_axis, df['cloture'], color='blue', lw=2)
    ax.plot(x_axis, df['MMA20'], color='black', lw=2)
    
    # Set Title & Show the Image
    ax.set_title('Bandes de Bollinger')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prix (USD)')
    ax.legend()
    plt.show();

def RSI(df,n) : 
    
    df['hausses'] = np.zeros(len(df))
    df['baisses'] = np.zeros(len(df))
    
    df['hausses'] = np.where((df['cloture'] - df['cloture'].shift())>0,df['cloture'] - df['cloture'].shift(), df['hausses'])
    df['baisses'] = np.where((df['cloture'] - df['cloture'].shift())<0,df['cloture'] - df['cloture'].shift(), df['baisses'])
    
    # H : moyenne mobile des hausses sur les n derniers jours
    df['H'] = df['hausses'].ewm(span=n,adjust=False).mean()
    # B : moyenne mobile des baisses sur les n derniers jours
    df['B'] = abs(df['baisses'].ewm(span=n,adjust=False).mean())
    
    # RSI = (H/(H+B))*100
    df['RSI'] = (df['H']/(df['H'] + df['B']))*100

    df = df.drop(['hausses','baisses','H','B'], axis=1)
    
    return df

def strategieRSI(df) :
    
    df['position'] = np.zeros(len(df))
    
    idx = np.argwhere(np.diff(np.sign(df['RSI'] - 70)) != 0).reshape(-1) + 0
    idx2 = np.argwhere(np.diff(np.sign(df['RSI'] - 30)) != 0).reshape(-1) + 0
    
    df['position'][idx] = np.where(df['RSI'][idx]>70, 'vente', df['position'][idx]) 
    df['position'][idx2] = np.where(df['RSI'][idx2]<30, 'achat', df['position'][idx2])
    """
    df['position'] = np.where(df['RSI']>70, 'vente', df['position'])
    df['position'] = np.where(df['RSI']<30, 'achat', df['position'])
    """
    return df
    
def testStrategie(df):
    
    budget = 1000
    budget += df[df['position']=='vente']['cloture'].sum()
    budget -= df[df['position']=='achat']['cloture'].sum()
    
    # Si le nombre d'achats est different du nombre de ventes il y a un probleme
    if (df.groupby('position').size()[2]-df.groupby('position').size()[3]) != 0 : 
        if (df.groupby('position').size()[2]-df.groupby('position').size()[3])<0 :
            budget -= df['cloture'][len(df)-1]*(df.groupby('position').size()[3]-df.groupby('position').size()[2])
        else :
            budget += df['cloture'][len(df)-1]*(df.groupby('position').size()[2]-df.groupby('position').size()[3])
    
    return budget


if __name__ == '__main__':

    code = 'IBM-1'
    
    #Test de la fonction cours :
    df = cours(code)
    
    # Test de la fonction ATR :
    df = ATR(df)
    
    # Test de la fonction MM :
    df = MM(df,code)

    # Test de la fonction MACD :
    df = MACD(df)
    
    # Test de la fonction RSI :
    df = RSI(df, 25)
    
    # Test de la fonction bollinger :
    bollinger(df)
    
    # PLT.STYLE A CHANGER