from csv import reader
from data_element import DataElement
from random import randint
from pickle import dump
file = open(r"D:\MachineLearning\Datasets\Multi_Label_dataset\train.csv", 'r', encoding="utf-8")
lines = reader(file)

categories = ["Action","Adventure","Animation","Biography","Comedy","Crime","Documentary","Drama",
              "Family","Fantasy","History","Horror","Music","Musical","Mystery","N/A","News",
              "Reality-TV","Romance","Sci-Fi","Short,Sport","Thriller","War","Western"]

next(lines)

data = []

for line in lines:
    data.append(DataElement(line[0], [int(x) for x in line[2:]]))

print(data)

print("Categories : ", len(categories))

file.close()

file = open("dataset.data", 'wb')
dump(data, file)
file.close()

file = open("labels.data", 'wb')
dump(categories, file)
file.close()

train = []
val = []
test = []

for x in data:
    p = randint(0, 100)

    if p <= 10:
        test.append(x)
    else:
        if p <= 25:
            val.append(x)
            if p <= 15:
               train.append(x)
        else:
            train.append(x)


file = open("train.data", 'wb')
dump(train, file)
file.close()

file = open("val.data", 'wb')
dump(val, file)
file.close()

file = open("test.data", 'wb')
dump(test, file)
file.close()

print("All : ", len(data))
print("Train : ", len(train))
print("Val : ", len(val))
print("Test : ", len(test))

print("Only Val : ", len([x for x in val if x not in train]))