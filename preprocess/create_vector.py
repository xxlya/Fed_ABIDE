import numpy as np
import deepdish as dd
import param
import csv
import os
import multiprocessing


def create_vector(site_folder):
    site, num = site_folder
    file_dir = fold + '/' + site
    file = list(os.walk(file_dir))[0][-1][:]
    label = []
    data = []
    id = []
    rows = {}  # label: rows = ['UM_1_0050272':1 ...]
    with open(fold + '/abide_preprocessed.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i in reader:
            if list(i[0].split(','))[5] in ['UM_1', 'NYU', 'USM', 'UCLA_1']:
                name, lab = list(i[0].split(','))[6:8]
                lab = int(lab) % 2
                rows[name] = lab
    for filename in file:
        tmp = dd.io.load(fold + "//{}//{}".format(site, filename))
        tri = np.triu(tmp, 1).reshape(-1)
        tri = tri[tri != 0]
        tri[tri < 0] = 0
        data.append(tri)
        label.append(int(rows[filename[:num]]) % 2)
        id.append(filename)
    data = np.array(data)
    label = np.array(label, dtype=np.int32)
    id = np.array(id)
    dataset = {'data': data, 'label': label, 'id': id}
    dd.io.save(fold + '/{}.h5'.format(site), dataset)


np.random.seed(5)
fold = os.getcwd()
cores = 4 if multiprocessing.cpu_count() >= 4 else multiprocessing.cpu_count()
pool = multiprocessing.Pool(cores)
pool.map(create_vector, param.SITE_FOLDER)

