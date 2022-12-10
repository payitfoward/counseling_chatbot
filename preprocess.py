import pandas as pd
import re
from sklearn.model_selection import train_test_split

data = pd.read_csv("./data/wellness.csv")

def cleantext(data):
    text = kor.sub(' ',str(data.유저[i]))
    text = ' '.join(text.split())
    return text

train, test = train_test_split(data[["구분","유저"]], test_size=0.1,shuffle=True, stratify=data.구분)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

idxlist = []
for i in range(len(train.구분)):
    idxlist.append(str(i) + '\ttrain\t' + str(train.구분[i]))
    
for i in range(len(test.구분)):
    idxlist.append(str(i) + '\ttest\t' + str(test.구분[i]))
    
f = open('data/wellness.txt', 'w', encoding="utf-8")
f.write('\n'.join(idxlist))
f.close()

corpuslist = []
kor = re.compile('[^ 一-龥a-zA-Z가-힣+]')
for i in range(len(train.유저)):
    corpuslist.append(cleantext(train))
    
for i in range(len(test.유저)):
    corpuslist.append(cleantext(test))
    
f = open('data/corpus/wellness.txt', 'w', encoding="utf-8")
f.write('\n'.join(corpuslist))
f.close()