import os
import param
import multiprocessing
import numpy as np
import deepdish as dd
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting


def make_correlation_matrix(line, site, filename):
    seq_len = param.WINDOW_SIZE
    time_series = []
    # good = bad = 0
    n = len(line) - seq_len + 1

    for j in range(n):
        lst = []
        for i in line[j: j + seq_len]:
            lst.append(np.array(list(map(float, i.split()))))
        time_series.append(np.array(lst))
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([time_series[j]])[0]
        fisher = np.arctanh(correlation_matrix)
        np.fill_diagonal(fisher, 0)
        dd.io.save(folder + '/{}_correlation_matrix//{}_{}.h5'.format(site, filename, j), fisher)

        # # check whether there are lines all 0 in this subject
        # for i in range(num_region):
        #     # if np.all(correlation_matrix.sum(axis=0)[82] == 1):
        #     if np.all(fisher[i] == 0):
        #         bad += 1
        #         bad_lst.append("{}".format(filename))
        #         break
        # if i == (num_region - 1):
        #     good += 1
        #     good_lst.append("{}".format(filename))
        # # plot matrix heat map
        # plotting.plot_matrix(fisher, figure=(10, 8), labels=labels, vmax=0.8, vmin=-0.8, reorder=True)


def truncation(site):
    file_dir = folder + '/' + site
    file = list(os.walk(file_dir))[0][-1][:]
    # bad_lst = good_lst = []
    for filename in file:
        f = open(folder + "//{}//{}".format(site, filename))
        lines = f.readlines()[1:]
        make_correlation_matrix(lines, site, filename)
        f.close()


num_region = param.HO_NUM_REGION
labels = [str(i) for i in range(num_region)]
folder = os.getcwd()
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(cores)
pool.map(truncation, param.SITE)


# print(len(bad_lst), len(good_lst))
# bad_set = set(bad_lst)
# for i in bad_set:
#     print("rm {}/{}".format(site, i))
# plotting.show()