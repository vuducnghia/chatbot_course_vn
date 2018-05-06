import json

with open('data/data.json') as json_data:
    mang = json.load(json_data)
khoavien = []
mahocphan =[]
tenhocphan =[]
malop =[]
for i in mang['features']:
    khoavien.append(i['Khoa viện'])
    mahocphan.append(i['Mã HP'])
    tenhocphan.append(i['Tên HP'])
    malop.append(i['Mã lớp'])
    print(i['Mã HP'])
    print(i['Tên HP'])

print(len(khoavien))
khoavien = set(khoavien)
mahocphan = set(mahocphan)
tenhocphan = set(tenhocphan)
malop = set(malop)

print(len(khoavien))
print(len(mahocphan))
print(len(tenhocphan))
print(len(malop))