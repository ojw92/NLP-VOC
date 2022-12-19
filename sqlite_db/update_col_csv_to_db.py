# 20221115
# Updates 'galaxys10' table from 'ivoc.db' file with the 'Flairs' column of S22_links_3.csv
# For data points where galaxys10.flair has NULL value & post_detail_link matches the links given in 'no_flair_links2.txt'
    # the rows of VOC data 'S22_links_3.csv' were collected in the same order of 'no_flair_links2.txt', so it works


import csv, sqlite3
import codecs
import pandas as pd

con = sqlite3.connect("C:/Users/jinwoo1.oh/Desktop/Scrapers_test/ivoc.db") # change to 'sqlite:///your_filename.db' # C:/Users/jinwoo1.oh/Desktop/Scrapers_test/
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS galaxys10 (post_id text PRIMARY KEY,cc integer,title text,description text,sync text,post_detail_link text,sync_time text,last_post_time text, started_by_time text, flair);")

# S22_links_3.csv
with open('S22_links_3.csv', 'rt', encoding='utf-8') as mycsv: # `with` statement available in 2.5+
        # 'latin-1' cus of "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe7 in position 34: invalid continuation byte"
        # 'rb' cus of "_csv.Error: line contains NULL byte"
    # csv.DictReader uses first line in file for column headings by default
    # reader = csv.reader(mycsv)
    if '\0' in open('S22_links_3.csv', 'rt', encoding='utf-8').read():
        print ("you have null bytes in your input file :(")
        input()
    else:
        print ("you don't have null bytes in your input file :)")
        input()


    # https://stackoverflow.com/questions/7894856/line-contains-null-byte-in-csv-reader-python
    dr = csv.DictReader((x.replace('\0', '') for x in mycsv), delimiter = ',')   # comma is default delimiter
    #line = next(dr)   # skips a line
    #print(len(line))
    #print(line.keys())      # use this to check columns names. keys are broken
    #input()
    # index names must match column names in file
    to_db = [(i['Flair']) for i in dr]


# save .txt file of list of hyperlinks as a Python list
my_file = open("no_flair_links2.txt", "r")
my_data = my_file.read()
links_list = my_data.split("\n")
my_file.close()


for i in range(len(to_db)):
    cur.execute("UPDATE galaxys10 SET flair=? WHERE galaxys10.flair IS NULL AND post_detail_link=?", (to_db[i], links_list[i]))
con.commit()
con.close()
