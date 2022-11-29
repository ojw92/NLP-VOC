# scrapes & saves up to 1000 posts (Reddit's post listing limit) of /new, /hot, etc of any Subreddit

import requests
import pandas as pd
import time
from datetime import datetime

CLIENT_ID = '_KZHz2OTsb1LfGG4ec_ZYQ'
SECRET_KEY = 'qr6On5RtyXyD9lwmTLe-smqAmS4yPQ'

auth = requests.auth.HTTPBasicAuth(CLIENT_ID, SECRET_KEY)

with open('pw.txt', 'r') as f:
    pw = f.read()

data = {
    'grant_type' : 'password',
    'username' : 'seavoc',
    'password' : pw
}

headers = {'User-Agent' : 'MyAPI/0.0.1'}

res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth = auth, data = data, headers = headers)

TOKEN = res.json()['access_token']
headers = {**headers, **{'Authorization' : f'bearer {TOKEN}'}}
    # headers['Authorization'] = f'bearer {TOKEN}'


# ===================== separate to another block =======================


# access desired pages on target subreddit
# requests.get('https://oauth.reddit.com/api/v1/me', headers = headers).json()
res = requests.get('https://oauth.reddit.com/r/S22Ultra/new',
                headers = headers, params = {'limit' : '100', 'after' : 't2_xu2j1b'},
        # https://www.reddit.com/r/S22Ultra/comments/xd9ysj/are_you_upgrading_to_iphone_14_pro_or_pro_max/
                )
#res.json()     # newer post on top

df = pd.DataFrame()

for post in res.json()['data']['children']:
    df = df.append({
        '1 Title' : post['data']['title'],
        '2 Content' : post['data']['selftext'],
        '3 created_time' : datetime.fromtimestamp(post['data']['created_utc']),
        '4 Flair' : post['data']['author_flair_text'],
        '5 name' : post['data']['name'],  # name = kind + '_' + id
        '6 ' : post['kind'],
        '7  ' : post['data']['id'],
    }, ignore_index = True)

print(df)
#res.json()['data']['children'][0]['data']

# after one loop of 100 entries are catalogued, we take the latest value of 'post' and get
# post['data']['name'], and put this 


# ======================== separate to yet another block ==============================

for i in range(0,50):
    res = requests.get('https://oauth.reddit.com/r/S22Ultra/new',
                headers = headers, params = {'limit' : '100',
                'after' : post['kind'] + '_' + post['data']['id']},
    # for any subreddit & any of /hot,/new,/top, etc, you can only go back 1000 posts
                )
    
    time.sleep(3.5)

    for post in res.json()['data']['children']:
        df = df.append({
            '1 Title' : post['data']['title'],
            '2 Content' : post['data']['selftext'],
            '3 created_time' : datetime.fromtimestamp(post['data']['created_utc']),
            '4 Flair' : post['data']['author_flair_text'],
            '5 name' : post['data']['name'],  # name = kind + '_' + id
            '6 kind' : post['kind'],
            '7 id' : post['data']['id'],
        }, ignore_index = True)
    

print(df)



# check most currently saved post id (oldest post) and save the data in current dataframe to csv
# post['data']['name']
# df.to_csv('S22Ultra_new_25times_start_t2_xu2j1b_end_t3_xu2j1b.csv')
