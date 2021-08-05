import openpyxl

theFile = openpyxl.load_workbook('example.xlsx')
arr = theFile.sheetnames
# print(arr[0])
currentSheet = theFile[arr[0]]
print(currentSheet['B4'].value)


def listToString(s):
    str1 = ""
    for ele in s:
        str1 += str(ele)
        str1 += " "
    return str1


array_for_peridotite = [2, 6, 0, 6, 2, 0, 0, 2]
array_for_mafic = [1, 9, 0, 9, 2, 0, 0, 9]
array_for_transitional = [0, 1, 2, 3]
# for row in range(0, len(sample_array)):
#     ws.append([row])
# print(currentSheet['B4'].value)
currentSheet['L1'] = "Peridotite"
currentSheet['M1'] = "Mafic"
currentSheet['N1'] = "Transitional"
# currentSheet['M2'] = listToString(sample_array)
marker_row = ""
for i in range(0, len(array_for_peridotite)):
    current_column = "L"
    current_row = str(i+2)
    current_location = current_column + current_row
    currentSheet[current_location] = array_for_peridotite[i]
    marker_row = current_row

for i in range(0, len(array_for_mafic)):
    current_column = "M"
    current_row = str(i+2)
    current_location = current_column + current_row
    currentSheet[current_location] = array_for_mafic[i]
    marker_row = current_row

for i in range(0, len(array_for_transitional)):
    current_column = "N"
    current_row = str(i+2)
    current_location = current_column + current_row
    currentSheet[current_location] = array_for_transitional[i]
    marker_row = current_row

marker_location = "L"+str(int(marker_row)+1)
theFile.save("output.xlsx")
