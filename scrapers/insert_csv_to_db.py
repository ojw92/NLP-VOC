import csv, sqlite3
import codecs

con = sqlite3.connect("C:/Users/jinwoo1.oh/Desktop/Scrapers_test/ivoc.db")  # directory for my .db file
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS S22Ultra ('Index','Title','Content','created_time','Flair','name','kind','id','media_only','media','is_video','view_count','edited','likes','score','thumbnail','category','ups','downs','upvote_ratio');")  # my column names

# Do this for S22Ultra.csv & GalaxyS22.csv
with open('GalaxyS22.csv', 'rt', encoding='utf-8') as mycsv: # `with` statement available in 2.5+
        # 'latin-1' cus of "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe7 in position 34: invalid continuation byte"
        # 'rb' cus of "_csv.Error: line contains NULL byte"
    # csv.DictReader uses first line in file for column headings by default
    # reader = csv.reader(mycsv)
    if '\0' in open('GalaxyS22.csv', 'rt', encoding='utf-8').read():
        print ("you have null bytes in your input file :(")
        input()
    else:
        print ("you don't have null bytes in your input file :)")
        input()


    # https://stackoverflow.com/questions/7894856/line-contains-null-byte-in-csv-reader-python
    dr = csv.DictReader((x.replace('\0', '') for x in mycsv), delimiter = '\t')   # comma is default delimiter
    #line = next(dr)   # skips a line
    #print(len(line))
    #print(line.keys())      # use this to check columns names. keys are broken
    #input()
    # index names must match column names in file
    to_db = [(i[''], i['Title'], i['Content'], i['created_time'], i['Flair'],
              i['name'], i['kind'], i['id'], i['media_only'], i['media'],
              i['is_video'], i['view_count'], i['edited'], i['likes'],
              i['score'], i['thumbnail'], i['category'], i['ups'],
              i['downs'], i['upvote_ratio']) for i in dr] # for i in dr

cur.executemany("INSERT INTO S22Ultra ('Index','Title','Content','created_time','Flair','name','kind','id','media_only','media','is_video','view_count','edited','likes','score','thumbnail','category','ups','downs','upvote_ratio') VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", to_db)
con.commit()
con.close()


