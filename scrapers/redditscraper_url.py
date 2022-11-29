# opens & scrapes info from a list of S22 issues from Reddit (r/S22Ultra & r/GalaxyS22)
# import output file to sqlite db, and use it to update galaxys10 table by filling rows with NULL in "Flair" column

import requests
import pandas as pd
import time
from datetime import datetime
from timeit import default_timer as timer
start = timer()

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
url = pd.read_csv('no_flair_links.txt', sep='\n', header=None)     # list of hyperlinks
df = pd.DataFrame()

for i in range(len(url)):
    res = requests.get('https://oauth' + url[0][i][11:] ,  headers = headers)
    time.sleep(1.1)     # Reddit allows 60 requests/minute

    # res.json() gets returned as a list of 1 dictionary for some reason
    for post in res.json()[0]['data']['children']:
        df = df.append({
            'Title' : post['data']['title'],
            'Content' : post['data']['selftext'],
            'created_time' : datetime.fromtimestamp(post['data']['created_utc']).strftime("%m/%d/%Y %H:%M:%S"),
            'Flair' : post['data']['author_flair_text'],
            'name' : post['data']['name'],  # name = kind + '_' + id
            'kind' : post['kind'],
            'id' : post['data']['id'],
            # '8 post_hint' : post['data']['post_hint'],    # not all posts have post_hint
            'media_only' : post['data']['media_only'],
            'media' : post['data']['media'],
            'is_video' : post['data']['is_video'],
            'view_count' : post['data']['view_count'],
            'edited' : post['data']['edited'],
            'likes' : post['data']['likes'],
            'score' : post['data']['score'],
            'thumbnail' : post['data']['thumbnail'],
            'category' : post['data']['category'],
            'ups' : post['data']['ups'],
            'downs' : post['data']['downs'],
            'upvote_ratio' : post['data']['upvote_ratio'],

        }, ignore_index = True)

print(df)

df.to_csv('GalaxyS22_02-04.csv')


end = timer()
print("%4f minutes have passed" % float((end-start)/60))
