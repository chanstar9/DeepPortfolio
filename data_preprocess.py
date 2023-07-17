import numpy as np
import pandas as pd

from column import *

excel_file = pd.ExcelFile('data/Deep_Portfolio_data.xlsx')

# company
raw_companies = excel_file.parse('company', skiprows=8)

raw_companies = raw_companies.rename(columns={
    'Symbol': 'code',
    'Symbol Name': 'name',
})

# Save symbol names and item names.
names = raw_companies.drop_duplicates(subset='code', keep='last').loc[:, ['code', 'name']]
names = names.set_index('code')
item_name_num = len(raw_companies.loc[:1000, 'Item Name '].unique())
item_names = raw_companies.loc[:item_name_num - 1, 'Item Name ']

# Remove unnecessary columns, for example, Symbol, Kind, Item, Item Name, Frequency
raw_companies = raw_companies.drop(columns=['Kind', 'name', 'Item Name ', 'Item', 'Frequency'])

# Melt every items.
melted_companies = pd.DataFrame(columns=['code', 'date'])
melted_companies = melted_companies.set_index(['code', 'date'])
for index, item_name in enumerate(item_names):
    # Melt raw_benchmark. Symbole name -> code, column names -> date
    item_companies = pd.melt(raw_companies.iloc[index::item_name_num, :], id_vars=['code'], var_name='date',
                             value_name=item_name)
    item_companies['date'] = pd.to_datetime(item_companies['date'], format='%Y-$m-%D')
    item_companies = item_companies.set_index(['code', 'date'])
    melted_companies = melted_companies.join(item_companies, how='outer')

RENAME = {'수익률(%)': 'return',
          '종가(원)': 'close_p',
          '상장주식수 (보통)(주)': 'listed_shares',
          '수정계수': 'adj_c',
          '최대주주등-보통주 수(Q)': 'ms_shares',
          '5%이상(10%미만)주주-보통주 수(Q)': 'over_20_quartile_shares',
          '10%이상(최대주주등제외)주주-보통주 수(Q)': 'over_10_quartile_shares',
          'FnGuide Industry Group 27': 'sector',
          '거래량(주)': 'volume'
          }
melted_companies = melted_companies.rename(columns=RENAME)
melted_companies.drop(columns=['수정계수 (현금배당반영)', '상장예정주식수 (보통)(주)', '거래소 업종', 'FnGuide Sector'], inplace=True)

# Add the names of company.
melted_companies = melted_companies.join(names)
melted_companies = melted_companies.reset_index()
melted_companies = melted_companies.sort_values(['code', 'date'])


# calculate market information
def zero_to_nan(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: np.where(x == 0, np.nan, x))


melted_companies.reset_index(inplace=True, drop=True)
melted_companies['mkt_cap'] = melted_companies['close_p'] * melted_companies['listed_shares']
melted_companies['adj_c'] = melted_companies.groupby('code').apply(
    lambda x: x['adj_c'][::-1].cumprod()[::-1].shift(-1).fillna(1)).reset_index(drop=True)
melted_companies['close_p'] = melted_companies.groupby('code').apply(
    lambda x: x['close_p'] / zero_to_nan(x['adj_c'])).reset_index(drop=True)
melted_companies['return'] = melted_companies.groupby('code').apply(
    lambda x: (x['close_p'].shift(-1) - x['close_p']) / zero_to_nan(x['close_p'])).reset_index(drop=True)
melted_companies['os_shares'] = melted_companies['listed_shares'] - \
                                (melted_companies['ms_shares'] + melted_companies['over_20_quartile_shares'] +
                                 melted_companies['over_10_quartile_shares'])

# index
raw_indices = excel_file.parse('index', skiprows=8)
raw_indices.rename(columns={'Symbol Name': 'name'}, inplace=True)
item_name_num = len(raw_indices.loc[:1000, 'Item Name '].unique())
item_names = raw_indices.loc[:item_name_num - 1, 'Item Name ']
raw_indices = raw_indices.drop(columns=['Symbol', 'Kind', 'Item', 'Item Name ', 'Frequency'])

melted_indices = pd.DataFrame(columns=['name', 'date'])
melted_indices = melted_indices.set_index(['name', 'date'])
for index, item_name in enumerate(item_names):
    # Melt raw_benchmark. Symbole name -> code, column names -> date
    item_indices = pd.melt(raw_indices.iloc[index::item_name_num, :], id_vars=['name'], var_name='date',
                           value_name=item_name)
    item_indices['date'] = pd.to_datetime(item_indices['date'], format='%Y-$m-%D')
    item_indices = item_indices.set_index(['name', 'date'])
    melted_indices = melted_indices.join(item_indices, how='outer')

melted_indices.rename(columns={'수익률(%)': 'return', '종가지수(포인트)': 'close_p'}, inplace=True)
melted_indices['return'] *= 0.01
melted_indices.reset_index(inplace=True)
melted_indices = melted_indices.sort_values(['name', 'date'])
melted_indices = melted_indices[melted_indices[DATE] < '2020']

#
melted_companies.to_csv('data/company_total.csv', index=False, encoding='utf-8')
melted_indices.to_csv('data/index_total.csv', index=False, encoding='utf-8')
