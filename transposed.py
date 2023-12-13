import os
import numpy as np
import pandas as pd

def process_load(path1, path2):

    if path1[-4:] == 'xlsx':
        df1 = pd.read_excel(path1, sheet_name=None)
        combined_df = pd.concat(df1.values(), ignore_index=True)
        target_columns = combined_df[['SID', 'DATE_DISPENSED', 'FDC', 'FDC2', 'NRTI', 'INTEGRASE', "NNRTI'", 'PI', 'BOOSTER', 'CCR5I', 'T20']]
        target_columns.columns = ['MRN', 'DATE_DISPENSED', 'FDC', 'FDC2', 'NRTI', 'INTEGRASE', "NNRTI'", 'PI', 'BOOSTER', 'CCR5I', 'T20']

    else:
        df1 = pd.read_csv(path1)
        target_columns = df1[['SID', 'DATE_DISPENSED', 'FDC', 'FDC2', 'NRTI', 'INTEGRASE', "NNRTI'", 'PI', 'BOOSTER', 'CCR5I', 'T20']]
        target_columns.columns = ['MRN', 'DATE_DISPENSED', 'FDC', 'FDC2', 'NRTI', 'INTEGRASE', "NNRTI'", 'PI', 'BOOSTER', 'CCR5I', 'T20']

    df2 = pd.read_csv(path2)
    df2 = df2[['MRN','DATE_DISPENSED', 'BOOSTER', 'CABENUVA', 'CCRI', 'FDC', 'FDC2', 'INTEGRASE', "NNRTI'", 'NRTI', 'OTHER', 'PI', 'T']]
    df2['DATE_DISPENSED'] = pd.to_datetime(df2['DATE_DISPENSED'])
    selected = df2[df2['DATE_DISPENSED'] > '2022-1-31']
    unmatch1 = [name for name in selected.columns if name not in target_columns.columns]
    unmatch2 = [name for name in target_columns.columns if name not in selected.columns]
    prepro1 = target_columns.copy()
    prepro2 = selected.copy()
    for name in unmatch1:
        prepro1.loc[:, name] = ''
    for name in unmatch2:
        prepro2.loc[:, name] = ''
    processed_df1 = prepro1[prepro2.columns].copy()
    processed_df2 = prepro2[prepro2.columns].copy()
    print(processed_df1.shape)
    print(processed_df2.shape)
    # combined = pd.concat([processed_df1,processed_df2], axis=0)
    # print(combined.shape)
    # print(combined.columns)
    # combined = combined.sort_values(by=['MRN', 'DATE_DISPENSED'])

######
    def inner_process(data_frame, column_name):
        data_frame['NRTI'] = data_frame['NRTI'].replace('VIREAD', 'TDF', regex=True)
        data_frame['NRTI'] = data_frame['NRTI'].replace('TENOFOVIR FUM', 'TDF', regex=True)
        data_frame['PI'] = data_frame['PI'].replace('PREZCOBIX', 'PREZCOB', regex=True)
        for column in column_name:
            data_frame[column] = data_frame[column].apply(
                lambda x: ' '.join(filter(lambda y: ((y != 'nan') and (y != 'na') and (y != 'n')), str(x).split())))

        final, current = [], {column: [[data_frame.loc[0, column]]] for column in column_name}
        current['MRN'], current['DATE_DISPENSED'] = [data_frame.loc[0, 'MRN']], [data_frame.loc[0, 'DATE_DISPENSED']]

        for index in range(1, len(data_frame)):
            current_date = pd.to_datetime(current['DATE_DISPENSED'][-1])
            # print(data_frame.loc[index, 'MRN'].dtype, ' check ', index, current['MRN'][-1].dtype)
            print(
                    data_frame.loc[index, 'DATE_DISPENSED'], current_date)
            if data_frame.loc[index, 'MRN'] == current['MRN'][-1] and (
                    data_frame.loc[index, 'DATE_DISPENSED'] - current_date).days <= 5:
                for name in column_name:
                    if current[name][-1][-1] != data_frame.loc[index, name]:
                        current[name][-1].append(data_frame.loc[index, name])
            else:
                for name in column_name:
                    current[name][-1] = ' '.join(sorted(filter(None, current[name][-1])))
                    current[name].append([data_frame.loc[index, name]])

                for name in ['MRN', 'DATE_DISPENSED']:
                    current[name].append(data_frame.loc[index, name])

        for name in column_name:
            current[name][-1] = ' '.join(sorted(filter(None, current[name][-1])))

        newdf = pd.DataFrame(current)
        return newdf

    processed_df1 = processed_df1.sort_values(by=['MRN', 'DATE_DISPENSED'])
    print(prepro2.columns)
    new_df_1 = inner_process(processed_df1, ['BOOSTER', 'CABENUVA', 'CCRI', 'FDC', 'FDC2',
           'INTEGRASE', "NNRTI'", 'NRTI', 'OTHER', 'PI', 'T', 'CCR5I', 'T20'])
    combined = pd.concat([new_df_1, processed_df2], axis=0)




    return combined

def pivot_transform(newdf, column_name):


    def filter_sort_and_concat(column):
        # Filter out non-string values, sort the remaining values, and concatenate them
        sorted_concatenated = ' '.join(sorted(filter(None, map(str,column.dropna()))))
        return sorted_concatenated

    newdf['COMBINED'] = newdf[column_name].apply(filter_sort_and_concat, axis= 1)
    # data_frame['COMBINED'] = np.apply_along_axis(
    #     lambda row: ' '.join(filter(None, map(str, row))),
    #     axis=1,
    #     arr=data_frame[column_name].values
    # )
    newdf['COMBINED'] = df['COMBINED'].apply(lambda x: ' '.join(filter(lambda y: ((y != 'nan') and (y != 'na') and (y != 'n')), str(x).split())))
    newdf['Month_Year'] = newdf['DATE_DISPENSED'].dt.strftime('%Y-%m')
    selected = newdf[newdf['DATE_DISPENSED'] > '2019-12-31']


    return selected
    #pivot_df = df.pivot_table(index='MRN', columns='Year_Month', aggfunc='size', fill_value=0)


if __name__ == '__main__':
    df = process_load('PREVIOUS_FINAL.xlsx', 'check_test.csv')
    NN = ['BOOSTER', 'CABENUVA', 'CCRI', 'FDC', 'FDC2',
           'INTEGRASE', "NNRTI'", 'NRTI', 'OTHER', 'PI', 'T', 'CCR5I', 'T20']
    new_df = pivot_transform(df, NN)
    new_df.to_csv('find.csv')
    pivot_df = new_df.pivot_table(index=['MRN', 'COMBINED'], columns='Month_Year', aggfunc='size', fill_value='')
    pivot_df.to_csv('test_version.csv')


