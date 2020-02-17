import numpy as np

from util_check import find_invalid_low, find_invalid_up
from util_cost import cal_diff_1_vec
from util_cost import n_people

# constants #
N_families = 5000
N_days = 100
# constants #

def move_family(ifamily, nfamily, day0, day1, assigned_day, family_on_day, occupancy):
    # assigned_day
    assigned_day[ifamily] = day1
    # family_on_day
    family_on_day[day0].remove(ifamily)
    family_on_day[day1].add(ifamily)
    # occupancy
    occupancy[day0] -= nfamily
    occupancy[day1] += nfamily
    occupancy[-1] = occupancy[-2]

def move_n_family(family_ids, family_sizes, day0s, day1s, assigned_day, family_on_day, occupancy):
    for ifamily, nfamily, day0, day1 in zip(family_ids, family_sizes, day0s, day1s):
        # assigned_day
        assigned_day[ifamily] = day1
        # family_on_day
        family_on_day[day0].remove(ifamily)
        family_on_day[day1].add(ifamily)
        # occupancy
        occupancy[day0] -= nfamily
        occupancy[day1] += nfamily
    occupancy[-1] = occupancy[-2]

def correct_invalid(assigned_day, family_on_day, occupancy, lower_occupancy=125, upper_occupancy=300):
    """ correct days invalid using greedy
    start with highest occupancy day, in term mv one family to lowest cost
    check again and follow the above process
    """
    
    # Correct lower bound
    invalid_low = find_invalid_low(occupancy, lower_occupancy=lower_occupancy)
    if len(invalid_low) != 0:
        print('correct invalid: find %d days below lower. correctting ...' % len(invalid_low))

    while len(invalid_low) != 0:
        print('Lower bound invalid emaining: %d' % len(invalid_low))
        invalid_low_list = [d for d, _ in invalid_low]
        invalid_low_set  = set(invalid_low_list)

        best_ifamily, best_nfamily, best_day0, best_day1, best_ediff = -1, -1, -1, -1, np.inf
        for day0 in range(1, N_days+1):
            if day0 in invalid_low_set:
                continue

            for ifamily in family_on_day[day0]: # ifamily
                nfamily = n_people[ifamily]
                if occupancy[day0]-nfamily < lower_occupancy:
                    continue

                ediffs = cal_diff_1_vec(ifamily, occupancy, day0, invalid_low_list)
                
                imin = ediffs.argmin()
                if ediffs[imin] < best_ediff:
                    best_ifamily, best_nfamily, best_day0, best_day1, best_ediff = \
                            ifamily, nfamily, day0, invalid_low_list[imin], ediffs[imin]
            
        move_family(best_ifamily, best_nfamily, best_day0, best_day1, assigned_day, family_on_day, occupancy)
        invalid_low = find_invalid_low(occupancy, lower_occupancy=lower_occupancy)
    
    # Correct upper bound
    invalid_up = find_invalid_up(occupancy, upper_occupancy=upper_occupancy)
    if len(invalid_up) != 0:
        print('correct invalid: find %d days above upper. correctting ...' % len(invalid_up))

    while len(invalid_up) != 0:
        print('Upper bound invalid emaining: %d' % len(invalid_up))
        invalid_up.sort(key=lambda x: x[1], reverse=True)
        
        for day0, _ in invalid_up: # day0

            best_ifamily, best_nfamily, best_day1, best_ediff = -1, -1, -1, np.inf
            for ifamily in family_on_day[day0]: # ifamily
                nfamily = n_people[ifamily]

                day1 = [d for d in range(1,N_days+1) if (occupancy[d]+nfamily) <= upper_occupancy]
                ediffs = cal_diff_1_vec(ifamily, occupancy, day0, day1)
                
                imin = ediffs.argmin()
                if ediffs[imin] < best_ediff:
                    best_ifamily, best_nfamily, best_day1, best_ediff = ifamily, nfamily, day1[imin], ediffs[imin]
            
            move_family(best_ifamily, best_nfamily, day0, best_day1, assigned_day, family_on_day, occupancy)
        
        invalid_up = find_invalid_up(occupancy, upper_occupancy=upper_occupancy)

    print('Upper bound correction finished.')


# !!! just keep it to make the code run. use correct_invalid above !!! #
def correct_invalid_up(assigned_day, family_on_day, occupancy, upper_occupancy=300):
    """ correct days above upper_occupancy using greedy
    start with highest occupancy day, in term mv one family to lowest cost
    check again and follow the above process
    """
    
    invalid_days = find_invalid_up(occupancy, upper_occupancy=upper_occupancy)
    if len(invalid_days) != 0:
        print('correct invalid: find %d invalid days to correct.' % len(invalid_days))

    while len(invalid_days) != 0:
        invalid_days.sort(key=lambda x: x[1], reverse=True)
        for day0, _ in invalid_days: # day0

            best_ifamily, best_nfamily, best_day1, best_ediff = -1, -1, -1, np.inf
            for ifamily in family_on_day[day0]: # ifamily
                nfamily = n_people[ifamily]

                day1 = [d for d in range(1,N_days+1) if (occupancy[d]+nfamily) <= upper_occupancy]
                ediffs = cal_diff_1_vec(ifamily, occupancy, day0, day1)
                
                imin = ediffs.argmin()
                if ediffs[imin] < best_ediff:
                    best_ifamily, best_nfamily, best_day1, best_ediff = ifamily, nfamily, day1[imin], ediffs[imin]
            
            move_family(best_ifamily, best_nfamily, day0, best_day1, assigned_day, family_on_day, occupancy)
        
        invalid_days = find_invalid_up(occupancy, upper_occupancy=upper_occupancy)

