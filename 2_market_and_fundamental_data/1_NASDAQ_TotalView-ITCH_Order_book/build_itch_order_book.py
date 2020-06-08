import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urljoin
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from struct import unpack
from collections import namedtuple, Counter
from datetime import timedelta
from time import time

# Set Data paths
data = '/Users/pengzhiyuan/PycharmProjects/machine_learning_for_algorithm_trading/data'
data_path = Path(data)
print(data_path)
itch_store = str(data_path / 'itch.h5')
order_book_store = data_path / 'order_book.h5'

# The FTP address, filename and corresponding date used in this example:
FTP_URL = 'ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/'
SOURCE_FILE = '03272019.NASDAQ_ITCH50.gz'


def may_be_download(url):
    """Download & unzip ITCH data if not yet available"""
    filename = data_path / url.split('/')[-1]
    print(filename)
    if not data_path.exists():
        print('Creating directory')
        data_path.mkdir()
    else:
        print('Directory exists')

    if not filename.exists():
        print('Downloading...', url)
        urlretrieve(url, filename)
    else:
        print('File exists')

    unzipped = data_path / (filename.stem + '.bin')
    if not (data_path / unzipped).exists():
        print('Unzipping to', unzipped)
        with gzip.open(str(filename), 'rb') as f_in:
            with open(unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print('File already unpacked')
    return unzipped


# This will download 5.1GB data that unzips to 12.9GB
file_name = may_be_download(urljoin(FTP_URL, SOURCE_FILE))
date = file_name.name.split('.')[0]
print(date)

# ITCH Format Settings
event_codes = {
    'O': 'Start of Messages',
    'S': 'Start of System Hours',
    'Q': 'Start of Market Hours',
    'M': 'End of Market Hours',
    'E': 'End of System Hours',
    'C': 'End of Messages'
}

encoding = {
    'primary_market_maker': {'Y': 1, 'N': 0},
    'printable': {'Y': 1, 'N': 0},
    'buy_sell_indicator': {'B': 1, 'S': -1},
    'cross_type': {'O': 0, 'C': 1, 'H': 2},
    'imbalance_direction':{'B': 0, 'S': 1, 'N': 0, 'O': -1}
}

formats = {
    ('integer', 2): 'H',
    ('integer', 4): 'I',
    ('integer', 6): '6s',
    ('integer', 8): 'Q',
    ('alpha', 1)  : 's',
    ('alpha', 2)  : '2s',
    ('alpha', 4)  : '4s',
    ('alpha', 8)  : '8s',
    ('price_4', 4): 'I',
    ('price_8', 8): 'Q',
}

# Create message specs for binary data parser
# Load Message Types

message_data = (pd.read_excel('message_types.xlsx',
                              sheet_name='message',
                              encoding='latin1')
                .sort_values('id')
                .drop('id', axis=1))

# Basic Cleaning

def clean_message_types(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df.value = df.value.str.strip()
    df.name = (df.name
               .str.strip() # remove whitespace
               .str.lower()
               .str.replace(' ', '_')
               .str.replace('-', '_')
               .str.replace('/', '_')
    )
    df.notes = df.notes.str.strip()
    df['message_type'] = df.loc[df.name == 'message_type', 'value']
    return df