import os
import pandas as pd
from LSTM import *

# load data that needed to be updated


def import_data(path):

    if path[-4:] == 'xlsx':

        df = pd.read_excel(path, sheet_name=None)
        combined_df = pd.concat(df.values(), ignore_index=True)
        print(combined_df.columns)
        # combined_df.drop(['TYPE'], axis = 1, inplace=True)

    else:
        combined_df = pd.read_csv(path)

    return combined_df

# load the dictionary to convert the drug name to art and type


def load_dict_base(path):

    df = pd.read_csv(path)

    return df

# clean dictionary base


def clean_data_dict(df, standard_name, col_name=None, to_all=True):

    df.columns = standard_name

    if to_all:

        df = df.applymap(
            lambda x: x.strip().upper() if isinstance(
                x, str) else x)

    else:
        for col in col_name:
            df[col] = df[col].str.strip().str.upper()

    return df


def create_convert(df):

    converter = {}

    for row in df.itertuples():

        converter[row.TRADE_NAME] = row.TYPE

    return converter


def clean_input(df, word_index, model):

    df.columns = [col.upper().strip() for col in df.columns]

    if 'ART' in df.columns:
        df['ART'] = df['ART'].str.upper().str.strip()
    else:
        features = sentences_to_indices(df, word_index, 8)

        def map_value(value):
            return index_output.get(value, value)

        mapped_array = np.vectorize(map_value)(np.argmax(loaded_model.predict(features), axis=1))
        df['ART'] = pd.Series(mapped_array)

    df['DATE_DISPENSED'] = pd.to_datetime(df['DATE_DISPENSED'])
    #####pending to delete
    selected = df[df['DATE_DISPENSED'] > '2022-02-01']

    return df


def create_new_col(data_frame, converter):


    ##pending to be fixed##
    def apply_converter(row):
        return pd.Series({'TYPE': converter.get(row['ART'], 'UNK')})

    if 'TYPE' in data_frame.columns:
        return data_frame

    new_columns = data_frame.apply(apply_converter, axis=1)

    # Concatenate the new columns to the original DataFrame
    df_1 = pd.concat([data_frame, new_columns], axis=1)
    df_sorted = df_1.sort_values(by=['MRN', 'DATE_DISPENSED'])

    return df_sorted


def pivot_transform(data_frame, column_name):

    r = pd.pivot_table(data_frame, values='ART', index=['MRN', 'DATE_DISPENSED'], columns=["TYPE"],
        aggfunc=lambda x: " ".join(sorted(set(x)))).fillna('').reset_index()
    print(r.columns)
    r['NRTI'] = r['NRTI'].replace('VIREAD', 'TDF', regex=True)
    r['NRTI'] = r['NRTI'].replace('TENOFOVIR FUM', 'TDF', regex=True)
    r['PI'] = r['PI'].replace('PREZCOBIX', 'PREZCOB', regex=True)
    # r['COMBINED'] = np.apply_along_axis(
    #     lambda row: ' '.join(filter(None, map(str, row))),
    #     axis=1,
    #     arr=r[column_name].values
    # )

    def filter_sort_and_concat(column):
        # Filter out non-string values, sort the remaining values, and concatenate them
        sorted_concatenated = ' '.join(sorted(filter(None, map(str,column.dropna()))))
        return sorted_concatenated

    r['COMBINED'] = r[column_name].apply(filter_sort_and_concat, axis= 1)

    return r

def date_merge(df, time_lapse, selected):


    r = pd.pivot_table(df, values='ART', index=['MRN', 'DATE_DISPENSED'], columns=["TYPE"],
        aggfunc=lambda x: " ".join(sorted(set(x)))).fillna('').reset_index()

    r['NRTI'] = r['NRTI'].replace('VIREAD', 'TDF', regex=True)
    r['NRTI'] = r['NRTI'].replace('TENOFOVIR FUM', 'TDF', regex=True)
    r['PI'] = r['PI'].replace('PREZCOBIX', 'PREZCOB', regex=True)

    final, current = [], {column: [[r.loc[0, column]]] for column in selected}
    current['MRN'], current['DATE_DISPENSED'] = [r.loc[0, 'MRN']], [r.loc[0, 'DATE_DISPENSED']]

    for index in range(1, len(r)):
        current_date = pd.to_datetime(current['DATE_DISPENSED'][-1])

        if r.loc[index, 'MRN'] == current['MRN'][-1] and (
                r.loc[index, 'DATE_DISPENSED'] - current_date).days <= time_lapse:
            for name in selected:
                if current[name][-1][-1] != r.loc[index, name]:
                    current[name][-1].append(r.loc[index, name])
        else:
            for name in selected:
                current[name][-1] = ' '.join(sorted(filter(None, current[name][-1])))
                current[name].append([r.loc[index, name]])

            for name in ['MRN', 'DATE_DISPENSED']:
                current[name].append(r.loc[index, name])

    for name in selected:
        current[name][-1] = ' '.join(sorted(filter(None, current[name][-1])))

    newdf = pd.DataFrame(current)

    # final, current = [], {column: [[r.loc[0,column]]] for column in ["FDC", "FDC2", "NRTI", "INTEGRASE", "NNRTI'", "PI", "BOOSTER", "CCRI", 'T','CABENUVA']}
    # current['MRN'] = [r.loc[0,'MRN']]
    # current['DATE_DISPENSED'] = [r.loc[0, 'DATE_DISPENSED']]
    # print(current)
    # for index in range(1,len(r)):
    #     current_date = pd.to_datetime(current['DATE_DISPENSED'][-1])
    #     if r.loc[index,'MRN'] == current['MRN'][-1] and (r.loc[index, 'DATE_DISPENSED'] - current_date).days <= time_lapse:
    #         for name in ["FDC", "FDC2", "NRTI", "INTEGRASE", "NNRTI'", "PI", "BOOSTER", "CCRI", 'T','CABENUVA']:
    #             if current[name][-1][-1] != r.loc[index,name]:
    #                 current[name][-1].append(r.loc[index,name])
    #     else:
    #         for name in ["FDC", "FDC2", "NRTI", "INTEGRASE", "NNRTI'", "PI", "BOOSTER", "CCRI", 'T','CABENUVA']:
    #             current[name][-1] = ' '.join(sorted(filter(None, current[name][-1])))
    #             current[name].append([r.loc[index,name]])
    #         for name in ['MRN', 'DATE_DISPENSED']:
    #             current[name].append(r.loc[index, name])
    # for name in ["FDC", "FDC2", "NRTI", "INTEGRASE", "NNRTI'", "PI", "BOOSTER", "CCRI", 'T','CABENUVA']:
    #     current[name][-1] = ' '.join(sorted(filter(None, current[name][-1])))
    # newdf = pd.DataFrame(current)

    def filter_sort_and_concat(column):
        # Filter out non-string values, sort the remaining values, and concatenate them
        sorted_concatenated = ' '.join(sorted(filter(None, map(str,column.dropna()))))
        return sorted_concatenated

    newdf['COMBINED'] = newdf[["FDC", "FDC2", "NRTI", "INTEGRASE", "NNRTI'", "PI", "BOOSTER", "CCRI", 'T','CABENUVA']].apply(filter_sort_and_concat, axis= 1)

    return newdf






if __name__ == '__main__':

    pd.options.display.max_colwidth = None

    df = import_data('MED_ORDER_PRE.xlsx')
    index_output, output_index, type_map = load_dicts_from_json("output_dicts.json")
    word_index, index_word, word_vector_map, word_vector = load_dicts_from_json("word_dicts.json")
    # loaded_model = tensorflow.keras.models.load_model('my_model.keras')
    loaded_model = {}
    # df.drop(['ART'], axis=1, inplace= True)

    new_df = clean_input(df, word_index, loaded_model)
    new_df2 = create_new_col(new_df, type_map)

    NN = ["FDC", "FDC2", "NRTI", "INTEGRASE", "NNRTI'", "OTHER", "PI", "BOOSTER", "CCRI", 'T','CABENUVA']
    NN_ORDER = ['FDC', 'FDC2', 'INJ', 'INSTI', 'NNRTI', 'NRTI',
       'OTHER', 'PI']

    # checktest = date_merge(new_df2, 5, NN)
    # checktest.to_csv('check_test_order.csv', index = False)

    pivoted = pivot_transform(new_df2, NN_ORDER)
    pivoted.to_csv('version2_order.csv', index=False)


    result = pivoted.groupby(['MRN', 'COMBINED', (pivoted['COMBINED'] != pivoted['COMBINED'].shift()).cumsum()]).agg(
        Start_Date=('DATE_DISPENSED', 'first'),
        End_Date=('DATE_DISPENSED', 'last'),
        Count=('DATE_DISPENSED', 'count')
    ).reset_index(level=[0,1]).sort_values(by=['MRN', 'Start_Date'])

    result['Record'] = (
            "MRN_: " + result['MRN'].astype(str) +
            " Combined_Values: " + result['COMBINED'].astype(str) +
            " Start_Date: " + result['Start_Date'].astype(str) +
            " End_Date: " + result['End_Date'].astype(str) +
            " Count: " + result['Count'].astype(str)
    )

    result["Month_Count"] = ((result["End_Date"] - result["Start_Date"]).dt.days / 30.5).round(2)
    result["Month_Count"] = result["Month_Count"].replace(0, 1)
    result["Count_per_month"] = (result["Count"].astype(float) / result["Month_Count"].astype(float)).round(2)
    result[['MRN', 'COMBINED', 'Start_Date', 'End_Date', 'Count',
            'Month_Count', 'Count_per_month', 'Record']].to_csv('version3_order.csv', index=False)

    # print(result.groupby("MRN")['COMBINED'].nunique())

