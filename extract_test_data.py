# 20221128
# Extract specific columns from galaxys10 table, add new columns of data to it, and save the dataframe as .csv file
# The code only extracts today's VOC before proper labeling (Y & N only, with no R). It is meant to be unlabeled test data for classification
# Only works if scraper collected at EST timezone and this program is run also at EST timezone. Can modify timezone from this program, if needed
    # 20230911 edit:

# Issue:
    # There may be potential conflict of duplicates - filtering via '%fold%' may show Fold5/4 issues, but also 'secure folder' issues of other models
    # This may pose a problem when updating predicted values on the database file. Should drop duplicates in the final test file

import csv, sqlite3
import codecs
import pandas as pd
import re
from datetime import datetime, date, timedelta
from timeit import default_timer as timer
from pytz import timezone
from datafunc import add_category
start = timer()


tz = timezone('EST')
onul = datetime.now(tz) - timedelta(days=0)                        # 2022-11-28 21:41:37.530500-05:00         (days=1) for yesterday's
onul_sql = str(onul)[:4] + str(onul)[5:7] + str(onul)[8:10]        # yyyymmdd to use for date filter in SQL


con = sqlite3.connect("C:/Users/jinwoo1.oh/Desktop/Scrapers_test/ivoc.db") # change to 'sqlite:///your_filename.db' # C:/Users/jinwoo1.oh/Desktop/Scrapers_test/
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS galaxys10 (post_id text PRIMARY KEY,cc integer,title text,description text,sync text,post_detail_link text,sync_time text,last_post_time text, started_by_time text, flair);")


models = ['s22', 's23', 'fold', 'flip', 'watch', 'tab']
df = pd.DataFrame()


def get_test(model, df2, onul_sql):

    print(f"\nAppending test data for {model}...")

    # filter out desired columns from galaxys10 table
    cur.execute("SELECT title,description,sync,flair,sync_time FROM galaxys10 WHERE post_detail_link like ? AND (sync like ? or sync like ? or sync like ?) AND (substr(sync_time, 7,4) || substr(sync_time, 1,2) || substr(sync_time, 4,2)) BETWEEN ? and ?;",
                    (f"%{model}%", '%r%', '%y%', '%n%', onul_sql, onul_sql)   # 20448 results when filtering by '2022' (up to 11/15), 18203 when filtering up to 11/15 (missing posts started in 2021), same on SQLite browser
                )
    filtered = cur.fetchall()       # list of 20448 tuples, each with 4 elements specified from SELECT
    df = pd.DataFrame(filtered, columns=['Title', 'Content', 'Class', 'Flair', 'Date'])


    # add issue category
    df, __ = add_category(df)


    # convert data format
    df['Date'] = df['Date'].map(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))   # consolidates to yyyy-mm-dd hh:mm regardless of formatting


    # filter out today's data
    test_lo = str(onul - timedelta(days=1))[:11] + "23:59:59"           # timedelta(days=2) for yesterday's data
    test_hi = '2029-12-31 23:59:59'
    df_test = df[(df['Date'] > test_lo) & (df['Date'] <= test_hi)]      # test set from today's date (sync_date on ivoc.db)

    
    # add a column to identify model
    df_test['Model'] = model

    print(f'Describe compiled test data for {model} :')
    print(df_test['Class'].value_counts())      # R: 590, N: 1655 (20221012 - 20221115)

    return df2.append(df_test, ignore_index=True)


# append rows of data for each model
for model in models:
    df = get_test(model, df, onul_sql)
print(f"\nShape of final df :", df.shape)


end = timer()
print("\n%4f seconds, %4f minutes elapsed" % (float(end-start), float((end-start)/60)))

input("Press Enter to save & exit :)")
df.to_csv(f'{onul_sql}_test.csv')


con.commit()
con.close()
