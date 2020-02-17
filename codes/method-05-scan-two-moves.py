"""
scan all families one by one and see if moving them to other date improves.
"""

import time
import numpy as np
import pandas as pd

from util_io import init, finalize, dump_conf
from util_cost import cal_total, cal_diff_1, cal_diff_n
from util_cost import n_people
from util_check import check_valid_all, check_valid_move, check_valid_move_n
from util_events import move_n_family

# constants #
N_families = 5000
N_days = 100
# constants #

# params #
#path_init_conf = '../output/answer_temp_350k.csv'
path_init_conf = '../output/m06-improved.csv'
path_dump_improved = '../output/m05-improved.csv'
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

family_ids = np.array([-1, -1], dtype='int16')
day0s = np.array([-1, -1], dtype='int16')
day1s = np.array([-1, -1], dtype='int16')
family_sizes = np.array([0, 0], dtype='int8')

while is_change:
    is_change= False
    iters += 1

    for ifamily in range(N_families):
        day0 = assigned_day[ifamily]
        # nfamily = n_people[ifamily]
        day0s[0] = day0
        family_ids[0] = ifamily
        family_sizes[0] = n_people[ifamily]

        best_ediff = 0.
        for day1 in range(N_days):
            if day1 == day0:
                continue
            day0s[1] = day1
            day1s[0] = day1
            for ifamily_2 in family_on_day[day1]:
                family_ids[1] = ifamily_2
                family_sizes[1] = n_people[ifamily_2]
                for day2 in range(N_days):
                    if day2 == day1:
                        continue
                    day1s[1] = day2
                    if not check_valid_move_n(occupancy, day0s, day1s, family_sizes):
                        continue
                    ediff = cal_diff_n(family_ids, occupancy, day0s, day1s, family_sizes)
                    if ediff < best_ediff:
                        best_ediff = ediff
                        best_family_ids = family_ids.copy()
                        best_family_sizes = family_sizes.copy()
                        best_day0s = day0s.copy()
                        best_day1s = day1s.copy()

        if best_ediff < 0.:
            print(
                'Got improvement from {} to {}. i={}'.format(etotal, etotal + best_ediff, ifamily)
            )
            move_n_family(
                best_family_ids, best_family_sizes, best_day0s, best_day1s,
                assigned_day, family_on_day, occupancy
            )
            dump_conf(assigned_day, path_dump_improved)
            etotal += best_ediff
            is_change = True

    if is_change:
        etotal = cal_total(assigned_day, occupancy)
    print('Iteraion %d finished.' % iters)

# finalize
print('Final score:', cal_total(assigned_day, occupancy))
print('Valid:', check_valid_all(occupancy))
print('Time spent:', time.time() - t0cpu)
finalize(assigned_day)
