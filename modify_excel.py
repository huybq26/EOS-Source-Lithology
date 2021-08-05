import openpyxl

theFile = openpyxl.load_workbook('example.xlsx')
arr = theFile.sheetnames
# print(arr[0])
currentSheet = theFile[arr[0]]
print(currentSheet['B4'].value)


def listToString(s):

    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += str(ele)
        str1 += " "

    # return string
    return str1


sample_array = [2, 6, 0, 6, 2, 0, 0, 2]
# for row in range(0, len(sample_array)):
#     ws.append([row])
# print(currentSheet['B4'].value)
currentSheet['L1'] = "Classification"
currentSheet['M1'] = "P/T Ratio"
currentSheet['L2'] = "Transitional"
# currentSheet['M2'] = listToString(sample_array)
marker_row = ""
for i in range(0, len(sample_array)):
    current_column = "M"
    current_row = str(i+2)
    current_location = current_column + current_row
    currentSheet[current_location] = sample_array[i]
    marker_row = current_row

marker_location = "L"+str(int(marker_row)+1)
currentSheet[marker_location] = "Di Ngu"
theFile.save("output.xlsx")
