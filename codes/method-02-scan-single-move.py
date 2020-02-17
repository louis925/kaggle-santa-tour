"""
scan all families one by one and see if moving them to other date improves.
"""

import time
import numpy as np
import pandas as pd

from util_io import init, finalize, dump_conf
from util_cost import cal_total, cal_diff_1
from util_cost import n_people
from util_check import check_valid_all, check_valid_move
from util_events import move_family

# constants #
N_families = 5000
N_days = 100
# constants #

# params #
#path_init_conf = '../output/answer_temp_350k.csv'
path_init_conf = '../output/m03-improved.csv'
path_dump_improved = '../output/m02-improved.csv'
# params #

# t0
t0cpu = time.time()

# init
assigned_day, family_on_day, occupancy = init(path_conf=path_init_conf)
etotal = cal_total(assigned_day, occupancy)
print('Init config cost:', etotal)

# init vars
is_change = True
iters = 0

while is_change:
    is_change= False
    iters += 1

    for ifamily in np.random.permutation(N_families):
        day0 = assigned_day[ifamily]
        nfamily = n_people[ifamily]

        best_ediff = 0
        best_day1 = None
        for day1 in np.random.permutation(N_days):
            if day1 == day0:
                continue
            if not check_valid_move(occupancy, day0, day1, nfamily):
                continue

            ediff = cal_diff_1(ifamily, occupancy, day0, day1)
            if ediff < best_ediff:
                best_ediff = ediff
                best_day1 = day1

        if best_day1 is not None:
            print('Got improvement from {} to {}.'.format(etotal, etotal + best_ediff))
            move_family(ifamily, nfamily, day0, best_day1, assigned_day, family_on_day, occupancy)
            dump_conf(assigned_day, path_dump_improved)
            etotal += best_ediff
            is_change = True

    print('Iteraion %d finished.' % iters)

# finalize
print('Final score:', cal_total(assigned_day, occupancy))
print('Time spent:', time.time()-t0cpu)
finalize(assigned_day)
