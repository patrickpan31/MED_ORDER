import pandas as pd
import numpy as np
import re

original_dic = pd.read_csv('NEWMCN.csv')

number = set(str(i) for i in range(1000))
special_word = set(i + 'MG/ML' for i in number)
stop_word = {"PO", "SODIUM", "DISOPROXIL", "SULFATE", "HCL", "MESYLATE", "POTASSIUM", "SC", "MG", "ML", "MG/ML", 'MGTAB', 'MLTAB'}
TRADE_NAME = []
TYPE = []
ART = []

def filter_out_pattern(text):
    pattern = r'\d+(?:-\d+)*'  # This regex pattern matches the required format
    return re.sub(pattern, '', text)

for work in original_dic['TOTAL']:
    work = filter_out_pattern(work)
    word = work.split()
    nword = ''
    for w in word[0:-2]:
        if w not in number and w not in special_word and w not in stop_word:
            nword += w + ' '
        else:
            break
    nword = nword.strip()
    if nword == 'ABACAVIR':
        ART.append(nword)
        TRADE_NAME.append('ZIAGEN')
        TYPE.append('NRTI')
        continue
    if nword == 'ATAZANAVIR':
        ART.append(nword)
        TRADE_NAME.append('REYATAZ')
        TYPE.append('PI')
        continue
    if nword == 'COBICISTAT':
        ART.append(nword)
        TRADE_NAME.append('TYBOST')
        TYPE.append('BOOSTER')
        continue
    if nword == 'DARUNAVIR':
        ART.append(nword)
        TRADE_NAME.append('PREZISTA')
        TYPE.append('PI')
        continue
    if nword == 'DOLUTEGRAVIR':
        ART.append(nword)
        TRADE_NAME.append('TIVICAY')
        TYPE.append('INEGRASE')
        continue
    if nword == 'DORAVIRINE':
        ART.append(nword)
        TRADE_NAME.append('PIFELTRO')
        TYPE.append('NNRTI')
        continue
    if nword == 'EFAVIRENZ':
        ART.append(nword)
        TRADE_NAME.append('SUSTIVA')
        TYPE.append('NNRTI')
        continue
    if nword == 'EMTRICITABINE':
        ART.append(nword)
        TRADE_NAME.append('EMTRIVA')
        TYPE.append('NRTI')
        continue
    if nword == 'ENFUVIRTIDE':
        ART.append(nword)
        TRADE_NAME.append('FUZEON')
        TYPE.append('OTHER')
        continue
    if nword == 'ETRAVIRINE':
        ART.append(nword)
        TRADE_NAME.append('INTELENCE')
        TYPE.append('NNRTI')
        continue
    if nword == 'INDINAVIR':
        ART.append(nword)
        TRADE_NAME.append('CRIXIVAN')
        TYPE.append('PI')
        continue
    if nword == 'LAMIVUDINE':
        ART.append(nword)
        TRADE_NAME.append('EPIVIR')
        TYPE.append('NRTI')
        continue
    if nword == 'LAMIVUDINE-TENOFOVIR':
        ART.append(nword)
        TRADE_NAME.append('CIMDUO')
        TYPE.append('NRTI')
        continue
    if nword == 'MARAVIROC':
        ART.append(nword)
        TRADE_NAME.append('SELZENTRY')
        TYPE.append('OTHER')
        continue
    if nword == 'NELFINAVIR':
        ART.append(nword)
        TRADE_NAME.append('VIRACEPT')
        TYPE.append('PI')
        continue
    if nword == 'NEVIRAPINE':
        ART.append(nword)
        TRADE_NAME.append('VIRAMUNE')
        TYPE.append('NNRTI')
        continue
    if nword == 'RALTEGRAVIR':
        ART.append(nword)
        TRADE_NAME.append('ISENTRESS')
        TYPE.append('INTEGRASE')
        continue
    if nword == 'RILPIVIRINE':
        ART.append(nword)
        TRADE_NAME.append('EDURANT')
        TYPE.append('NNRTO')
        continue
    if nword == 'RITONAVIR':
        ART.append(nword)
        TRADE_NAME.append('NORVIR')
        TYPE.append('BOOSTER')
        continue
    if nword == 'SAQUINAVIR':
        ART.append(nword)
        TRADE_NAME.append('INVIRASE')
        TYPE.append('PI')
        continue
    if nword == 'STAVUDINE':
        ART.append(nword)
        TRADE_NAME.append('ZERIT')
        TYPE.append('NRTI')
        continue
    if nword == 'TIPRANAVIR':
        ART.append(nword)
        TRADE_NAME.append('APTIVUS')
        TYPE.append('PI')
        continue
    if nword == 'ZIDOVUDINE':
        ART.append(nword)
        TRADE_NAME.append('RETROVIR')
        TYPE.append('NRTI')
        continue
    ART.append(nword)
    TRADE_NAME.append(word[-2])
    TYPE.append(word[-1])

for special in ["LAMIVUDINE-ABACAVIR", "ABACAVIR-LAMIVUDINE", "EFAVIRENZ/TENOFOVIR/EMTRICI"]:
    ART.append(special)
    if special == "EFAVIRENZ/TENOFOVIR/EMTRICI":
        TRADE_NAME.append("ATRIPLA")
        TYPE.append("FDC")
    else:
        TRADE_NAME.append("EPZICOM")
        TYPE.append("NRTI")

df = pd.DataFrame({'ART': ART, 'TRADE_NAME': TRADE_NAME, 'TYPE':TYPE})
# df = (original_dic
#       .assign(ART=original_dic['TOTAL'].str.split().str[0].str.strip().str.upper(),
#               TRADE_NAME=original_dic['TOTAL'].str.split().str[-2].str.strip().str.upper(),
#               TYPE=original_dic['TOTAL'].str.split().str[-1].str.strip().str.upper()))
# df = df.drop(['TOTAL'], axis=1)

new_dic = pd.read_csv('art.csv')
new_dic.columns = ['ART', 'TRADE_NAME', 'TYPE']

new_dic['ART'] = new_dic['ART'].str.strip().str.upper()
new_dic['TRADE_NAME'] = new_dic['TRADE_NAME'].str.strip().str.upper()
new_dic['TYPE'] = new_dic['TYPE'].str.strip().str.upper()

result = pd.concat([new_dic, df]).reset_index(drop=True)
result.drop_duplicates(inplace=True)

result.to_csv('new_art2.csv', index = False)

print(original_dic['TOTAL'].str.split().str[-2].str.strip().str.upper())

