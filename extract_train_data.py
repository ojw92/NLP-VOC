# 20221115
# Extract specific columns from galaxys10 table, add new columns of data to it, and save the dataframe as .csv file
# Instructions:
    # Change filtered date range to control train & validation data (Jan-mid Feb data not necessary)
    # 20230120 edit: use the whole thing as training data - validation data should be determined during analysis
    # 20230308 edit: this creates
    # 20230811 edit: specify title for s22; create s23 version
    # 20230911 edit: consolidate training data extraction algorithm for all models - only collecting for S23/S22, Fold5/4, Flip5/4, Watch6/5, TabS9 - and implement data augmentation for S22 & S23

# Note:
    # S23 released in the US on Feb 17, 2023
    # S22 released in the US on Feb 25, 2022.
    # Fold4 released in the US on Aug 26, 2022; Fold5 released Aug 11, 2023
        # Fold3 was reported last on March 30, 2023. For augmentation, convert all Fold3 data up to that point to 4 or 5
    # Flip4 released in the US on Aug 26, 2022; Flip5 released Aug 11, 2023
        # Flip3 was reported last on March 7, 2023. For augmentation, convert all Flip3 data up to that point to 4 or 5
    # Watch5 released in the US on Aug 26, 2022; Watch6 released Aug 11, 2023
        # Watch4 was reported last on Jan 18, 2023. For augmentation, convert all Watch4 data up to that point to 5 or 6
    # TabS8 released in the US on Feb 25, 2022; TabS9 released Aug 11, 2023
        # TabS7 was never reported, so data for 'tab' will have 'R' for only Tab8 & Tab9
    # For data augmentation, remove Fold5/Flip5/Watch6/TabS9 data from before Aug 8 (1st day of reporting these models).

# Issue:
    # There may be potential conflict of duplicates - filtering via '%fold%' may show Fold5/4 issues, but also 'secure folder' issues of other models

import csv, sqlite3
import codecs
import pandas as pd
import re
from datetime import datetime
from timeit import default_timer as timer


model = str(input("Which model to extract training data? [fold, flip, watch, tab, s23, s22] : ")).lower()
model_filter = str("%" + model + "%")
if model == "s23":
    start_date = str("20230000")
else:
    start_date = str("20220000")

start = timer()


con = sqlite3.connect("C:/Users/jinwoo1.oh/Desktop/Scrapers_test/ivoc.db") # change to 'sqlite:///your_filename.db' # C:/Users/jinwoo1.oh/Desktop/Scrapers_test/
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS galaxys10 (post_id text PRIMARY KEY,cc integer,title text,description text,sync text,post_detail_link text,sync_time text,last_post_time text, started_by_time text, flair);")


# filter out desired columns from galaxys10 table
cur.execute("SELECT title,description,sync,flair,sync_time FROM galaxys10 WHERE post_detail_link like ? AND (sync like ? or sync like ?) AND (substr(sync_time, 7,4) || substr(sync_time, 1,2) || substr(sync_time, 4,2)) BETWEEN ? and ?;",
                (model_filter, '%r%', '%n%', start_date, '20291231')   # S23 released in the US on Feb 17, 2023
            )
filtered = cur.fetchall()       # list of tuples, each with 4 elements specified from SELECT
df = pd.DataFrame(filtered, columns=['Title', 'Content', 'Class', 'Flair', 'Date'])



# Filter out S22 data to add to S23 training data

# S23 data by itself upon release is not enough data to train a model to classify S23 issues. Need to convert S22 issues into S23
# To remove tricky cases that mess up training data, such as people upgrading from S22 to S23, remove all posts from S22 data that already mention S23. Even after omission, there's plenty of training data.
# https://www.geeksforgeeks.org/how-to-drop-rows-that-contain-a-specific-string-in-pandas/
    # Use regex to write a simpler command
# filter out desired columns from galaxys10 table
def augment_data(model):
    aug_dict = {
            's23' : ['%s22%', 'S23|S 23', '(?i)S\s*22'], 
            's22' : ['%s23%', 'S22|S 22', '(?i)S\s*23'],          # 's22' doesn't really need data augmentation since there's already a lot
            # fold, flip, watch, & tab issues need fine-tuning of date filtration since there's an overlap of period where older models were still reported and where older models were no longer monitored
            }
    
    cur.execute("SELECT title,description,sync,flair,sync_time FROM galaxys10 WHERE post_detail_link like ? AND (sync like ? or sync like ?) AND (substr(sync_time, 7,4) || substr(sync_time, 1,2) || substr(sync_time, 4,2)) BETWEEN ? and ?;",
                    (aug_dict[model][0], '%r%', '%n%', '20220000', '20291231')   # S22 released in the US on Feb 25, 2022
                )
    filtered_aug = cur.fetchall()
    df_aug = pd.DataFrame(filtered_aug, columns=['Title', 'Content', 'Class', 'Flair', 'Date'])
    df_aug = df_aug[~df_aug['Title'].str.contains(aug_dict[model][1], flags=re.IGNORECASE, regex=True)    | \
                    ~df_aug['Content'].str.contains(aug_dict[model][1], flags=re.IGNORECASE, regex=True)]
    
    # Convert all mentions of other model to target model (ex. convert all "s22" to "s23" from s22 posts so they can be added to s23)
    df_aug = df_aug.replace(aug_dict[model][2], model, regex=True)
    print(df_aug.head(10))

    # print(df22.head(10))
    # add this "cleaned" S22 training data into S23 data via vertical concatenation
    return df_aug

# Only apply augmentation to S23 & S22 models for now
if model == 's23' or model == 's22':
    df_aug = augment_data(model)
    print(f"\nShape of {model} df:", df.shape)
    print(f"Shape of df_aug:", df_aug.shape)
    df = pd.concat([df_aug, df], axis=0)

print(f"\nShape of final {model} df after augmentation, if applicable:", df.shape)


# Apply 'add_category' module from my 'datafunc' package
print("Adding issue category. This may take up to 20 seconds...")
from datafunc import add_category
df, df2 = add_category(df)


df['Date'] = df['Date'].map(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M'))   # consolidates to yyyy-mm-dd hh:mm regardless of formatting

"""
# use below lines to set range of dates for train & validation data
train_cutoff = '2023-07-31 23:59:59'
test_cutoff = '2023-08-10 23:59:59'
# Important note: S23 US release date was Feb 17, 2023. Pre-sale arrivals & non-US models around that time might influence the data trends
# Since 20230308, there is more than a year's worth of data, so consider skipping 2 wks, and start training set from 20220304, a week after release
df_train = df[(df['Date'] >= '2023-01-01') & (df['Date'] <= train_cutoff)]             # train set from 1/1 - 10/11
df_test = df[(df['Date'] > train_cutoff) & (df['Date'] <= test_cutoff)]      # val set from 10/11 - present


print('describe train :')
print(df_train['Class'].value_counts())     # R: 3659, N: 14544 (20221011 cutoff)
print('describe test :')
print(df_test['Class'].value_counts())      # R: 590, N: 1655 (20221012 - 20221115)
"""

end = timer()
print("\n%4f minutes have passed" % float((end-start)/60))

print(f"\nSaving training data to {model}_total.csv...")
input("Press Enter to save & exit :)")
df.to_csv(f'train_{model}.csv')
#df_train.to_csv('S23_train.csv')
#df_test.to_csv('S23_val.csv')


con.commit()
con.close()
