# Takes test data, splits by date and creates .csv file for each

from datetime import datetime, timedelta
import pandas as pd
import os
MODEL_DIR = './test_data/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


def split_by_date(df22):
    fd = datetime.strptime(df22['Date'].iloc[0],'%m/%d/%Y %H:%M').date()
    ld = datetime.strptime(df22['Date'].iloc[-1],'%m/%d/%Y %H:%M').date()
    print('First date: ', fd)
    print('Last date: ', ld)
    print('Difference: ', (ld - fd).days, 'days')     # difference + 1 = number of test sets
    print((ld - fd).days + 1, 'files will be generated.')
    for i in range((ld - fd).days + 1 ):     # from 0 to difference
        if (fd + timedelta(days=i)).weekday() == 5 or (fd + timedelta(days=i)).weekday() == 6:  # exclude wknd
            continue
        df = df22.loc[df22.Date.apply(lambda x: datetime.strptime(x,'%m/%d/%Y %H:%M').date() ==
            (fd + timedelta(days=i)))]
        df = df.sort_values(by='Class', ascending=False)   # R at the top; easy to see R(green) points in scatterplot
        df.to_csv(f'./test_data/test_{(fd + timedelta(days=i)).month:02d}{(fd + timedelta(days=i)).day:02d}.csv',
            sep='\t')     # default is ',' as delimiter


df = pd.read_csv('S22_test20221115.csv', sep='\t', index_col=0)
split_by_date(df)


# creates a list of directories to each test file
test_files = []
directory = 'test_data'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        test_files.append(f)

test_files = sorted(test_files)
test_files




