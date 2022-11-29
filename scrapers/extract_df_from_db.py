# 20221115
# Extract specific columns from galaxys10 table, add new columns of data to it, and save the dataframe as .csv file


import csv, sqlite3
import codecs
import pandas as pd
import re
from datetime import datetime
from timeit import default_timer as timer
start = timer()

con = sqlite3.connect("C:/Users/jinwoo1.oh/Desktop/Scrapers_test/ivoc.db") # change to 'sqlite:///your_filename.db' # C:/Users/jinwoo1.oh/Desktop/Scrapers_test/
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS galaxys10 (post_id text PRIMARY KEY,cc integer,title text,description text,sync text,post_detail_link text,sync_time text,last_post_time text, started_by_time text, flair);")


# filter out desired columns from galaxys10 table
cur.execute("SELECT title,description,sync,flair,sync_time FROM galaxys10 WHERE post_detail_link like ? AND (sync like ? or sync like ?) AND (substr(sync_time, 7,4) || substr(sync_time, 1,2) || substr(sync_time, 4,2)) BETWEEN ? and ?;",
                ("%s22%", '%r%', '%n%', '20220000', '20221115')   # 20448 results when filtering by '2022' (up to 11/15), 18203 when filtering up to 11/15 (missing posts started in 2021), same on SQLite browser
            )
filtered = cur.fetchall()       # list of 20448 tuples, each with 4 elements specified from SELECT
df = pd.DataFrame(filtered, columns=['Title', 'Content', 'Class', 'Flair', 'Date'])


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

# print(df2.head())
# print(df2['Text'].str.contains("|".join(battery_list), case=False))
df['Thirdparty'] = df2['Text'].str.contains("|".join(thirdparty_list), case=False)
df['Display'] = df2['Text'].str.contains("|".join(display_list), case=False)
df['Battery'] = df2['Text'].str.contains("|".join(battery_list), case=False)
df['Camera'] = df2['Text'].str.contains("|".join(camera_list), case=False)
df['Noti'] = df2['Text'].str.contains("|".join(noti_list), case=False)
df['Connect'] = df2['Text'].str.contains("|".join(connect_list), case=False)
df['Messages'] = df2['Text'].str.contains("|".join(messages_list), case=False)

df['Date'] = df['Date'].map(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))   # consolidates to yyyy-mm-dd hh:mm regardless of formatting


train_cutoff = '2022-10-11 23:59:59'
test_cutoff = '2022-12-31 23:59:59'
df_train = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= train_cutoff)]             # train set from 1/1 - 10/11
df_test = df[(df['Date'] > train_cutoff) & (df['Date'] <= test_cutoff)]      # train set from 10/11 - present

print('describe train :')
print(df_train['Class'].value_counts())     # R: 3659, N: 14544 (20221011 cutoff)
print('describe test :')
print(df_test['Class'].value_counts())      # R: 590, N: 1655 (20221012 - 20221115)


end = timer()
print("%4f minutes have passed" % float((end-start)/60))

input("Press Enter to save & exit :)")
df.to_csv('S22_total.csv')
df_train.to_csv('S22_train.csv')
df_test.to_csv('S22_test.csv')


con.commit()
con.close()
