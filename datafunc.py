# 20230811
# Function file

import csv, sqlite3
import codecs
import pandas as pd
import re
from datetime import datetime



def add_category(df):

    # https://www.statology.org/can-only-use-str-accessor-with-string-values/
        # when you have a mix of integer & string to match, and get the "Can only use .str accessor with string values!" error
        # need to use .astype(str) before using the .str.contains(), .str.replace(), etc
    thirdparty_list = ['facebook', 'snapchat', 'youtube', 'instagram', 'reddit', 'amazon prime', 'cod',
                        'fb', 'spotify', 'netflix', 'zoom', 'discord', 'tik', 'whatsapp', 'twitter',
                        'genshin', 'game', 'dropbox', 'onedrive', 'twitch']
    display_list = ['display', 'screen', 'crack', 'scratch', 'protector', 'hz', 'scroll', 'touch', 'hdr',
                    'refresh rate', 'flicker', 'pixel', 'burn-in', 'burn in', 'tint', 'jump']
    battery_list = ['SoT', 'battery', 'usage', 'consum', 'drain', 'lasts', 'lasting', 'dies']
    camera_list = ['camera', 'shot', 'focus', 'blur', 'astrophotography', 'photo', 'video', 'saturat',
                    'shutter', 'selfie', 'record', 'lens', 'ultrawide', 'ultra wide', 'flash', 'slow-mo']
    noti_list = ['notif', 'vibrat', 'incoming', 'pop-up']
    connect_list = ['connect','network','mobile data','hotspot','esim','sim card','5g','4g','3g',
                    'signal', 'speed', 'internet', 'cellular', 'dual', 'reception', 'coverage']
        # 'data', 'service', 'sim' might catch wrong VOC
    messages_list = ['text', 'messag', 'whatsapp', 'RCS', 'MMS', 'chat', 'send', 'WhatsApp']


    df2 = df.replace(float("nan"), '', regex=True)
    df2 = df2.iloc[:,0] + " " + df2.iloc[:,1]      # slicing DataFrame makes it a Series
    df2 = pd.DataFrame(df2, columns= ['Text'])

    # Label True or False for issue category for each VOC
    # print(df2.head())
    # print(df2['Text'].str.contains("|".join(battery_list), case=False))
    df['Thirdparty'] = df2['Text'].str.contains("|".join(thirdparty_list), case=False)
    df['Display'] = df2['Text'].str.contains("|".join(display_list), case=False)
    df['Battery'] = df2['Text'].str.contains("|".join(battery_list), case=False)
    df['Camera'] = df2['Text'].str.contains("|".join(camera_list), case=False)
    df['Noti'] = df2['Text'].str.contains("|".join(noti_list), case=False)
    df['Connect'] = df2['Text'].str.contains("|".join(connect_list), case=False)
    df['Messages'] = df2['Text'].str.contains("|".join(messages_list), case=False)


    return df, df2
