"""
select one family and then put them into diff (valid) day.
if cost reduce, accept it; otherwise, accept by probability of exp(-cost_diff/temperature)
anneal in temp. and upper_occupancy.
NOTICE: upper_occupancy constraint may not valid at some point.
NOTICE: if found larger than upper_occupancy, move family no matter what.
"""

import numpy as np
import pandas as pd

from util_io import init, finalize, dump_conf
from util_cost import cal_total, cal_diff_n
from util_cost import n_people
from util_check import check_valid_all, check_valid_move_n, deep_check
from util_events import move_n_family, correct_invalid_up

# constants #
N_families = 5000
N_days = 100
# constants #

# params #
path_init_conf = '../output/m05-improved.csv'
path_dump_improved = '../output/m06-improved.csv' # lowest cost
path_dump_continue = '../output/m06-continue.csv' # current state

N_runs = 5
list_temp = [0.5 * t for t in range(10, 5, -1)]
list_upper_occupancy = [300 for i in range(5)]
list_N_iters = [10000000 for i in range(5)]

assert len(list_temp) == N_runs, 'size list_temp != N_runs: %d, %d' % (len(list_temp), N_runs)
assert len(list_upper_occupancy) == N_runs, 'size list_upper_occupancy != N_runs: %d, %d' % (len(list_upper_occupancy), N_runs)
assert len(list_N_iters) == N_runs, 'size list_N_iters != N_runs: %d, %d' % (len(list_N_iters), N_runs)

N_dumps = 5000000 # dump every iters
# params #

# init
print('list_temp:')
print(list_temp)
print('list_upper_occupancy:')
print(list_upper_occupancy)
print('list_N_iters:')
print(list_N_iters)

assigned_day, family_on_day, occupancy = init(path_conf=path_init_conf)
etotal_low = cal_total(assigned_day, occupancy)
print('Init config cost:', etotal_low)

# init vars
ifamilies = np.array([-1, -1], dtype='int16')
nfamilies = np.array([ 0,  0], dtype='int8')
day0s = np.array([-1, -1], dtype='int16')
day1s = np.array([-1, -1], dtype='int16')

for i_run in range(N_runs):
    # params set up
    temp = list_temp[i_run]
    upper_occupancy = list_upper_occupancy[i_run]
    N_iters = list_N_iters[i_run]

    # correct invalid days (if > upper_occ)
    correct_invalid_up(assigned_day, family_on_day, occupancy, upper_occupancy=upper_occupancy)
    deep_check(assigned_day, family_on_day, occupancy, upper_occupancy=upper_occupancy)

    # init vars
    etotal = cal_total(assigned_day, occupancy)
    is_change = False
   
    print('At temp = %.1f, upper_occ = %d, N_iters = %d. init cost: %.5f' % (temp, upper_occupancy, N_iters, etotal))

    for iters in range(N_iters):
        # dump conf
        if is_change and iters % N_dumps == 0:
            dump_conf(assigned_day, path_dump_continue)
            is_change = False

        # pick families (always move 1 to 2)
        ifamilies[0] = np.random.randint(N_families)
        ifamilies[1] = np.random.randint(N_families)
        while assigned_day[ifamilies[0]] == assigned_day[ifamilies[1]]:
            ifamilies[1] = np.random.randint(N_families)
        nfamilies[0] = n_people[ifamilies[0]]
        nfamilies[1] = n_people[ifamilies[1]]

        # pick days
        day0s[0] = assigned_day[ifamilies[0]]
        day0s[1] = assigned_day[ifamilies[1]]
        day1s[0] = assigned_day[ifamilies[1]]
        day21 = np.random.randint(1, N_days)
        day21 = day21 if day21 < assigned_day[ifamilies[1]] else day21 + 1
        day1s[1] = day21

        # check validity
        if not check_valid_move_n(occupancy, day0s, day1s, nfamilies, upper_occupancy=upper_occupancy):
            continue

        # determine acceptance
        ediff = cal_diff_n(ifamilies, occupancy, day0s, day1s, nfamilies)
        if ediff > 0 and np.random.random() >= np.exp(-ediff/temp):
            continue
        else:
            move_n_family(ifamilies, nfamilies, day0s, day1s, assigned_day, family_on_day, occupancy)
            is_change = True
    
        # update etotal and check improvement
        etotal = etotal + ediff
        if etotal < etotal_low:
            print('{}/{}. Got improvement from {} to {}.'.format(iters, N_iters, etotal_low, etotal))
            dump_conf(assigned_day, path_dump_improved)
            etotal_low = etotal

# finalize
print('Final score:', cal_total(assigned_day, occupancy))
finalize(assigned_day, path_dump=path_dump_continue)
