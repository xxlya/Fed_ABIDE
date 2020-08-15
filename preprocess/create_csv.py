import os
import param

lst = []
id = []

for site in param.SITE:
    folder = os.getcwd()
    file_dir = folder + '/' + site
    lst += list(os.walk(file_dir))[0][-1][:]

for file in lst:
    if file[:4] == 'UCLA':
        id.append(("UCLA_1", file.split("_rois")[0][-5:]))
    elif file[:3] == "USM":
        id.append(("USM", file.split("_rois")[0][-5:]))
    elif file[:3] == "NYU":
        id.append(("NYU", file.split("_rois")[0][-5:]))
    elif file[:2] == "UM":
        id.append(("UM_1", file.split("_rois")[0][-5:]))

with open("Phenotypic_V1_0b_preprocessed1.csv") as f:
    lines = f.readlines()

for line in lines:
    elements = line.split(",")
    if (elements[5], elements[4]) in id:
        print(line.split("\n")[0])
