import operator
from functools import reduce
from util_cost import n_people

# constants #
N_days = 100
# constants #

and_opt = operator.and_

def check_valid_all(occupancy, lower_occupancy=125, upper_occupancy=300):
    assert len(occupancy) == 102
    assert occupancy[-1] == occupancy[-2]
    occupancy = occupancy[1:-1]
    return occupancy.max() <= upper_occupancy and occupancy.min() >= lower_occupancy

def check_valid_move(occupancy, day0, day1, n, lower_occupancy=125, upper_occupancy=300):
    """ check valid for case when moving n people from day0 to day1 """
    return (occupancy[day0]-n) >= lower_occupancy and (occupancy[day1]+n) <= upper_occupancy

def check_valid_move_n(occupancy, day0s, day1s, family_sizes, lower_occupancy=125, upper_occupancy=300):
    """ check valid for case when moving multiple family from day0 to day1 """
    day_set = set(day0s) | set(day1s)
    occupancy_temp = {day: occupancy[day] for day in day_set}
    for day0, day1, n in zip(day0s, day1s, family_sizes):
        occupancy_temp[day0] -= n
        occupancy_temp[day1] += n
    return reduce(and_opt,
        [lower_occupancy <= occ <= upper_occupancy for occ in occupancy_temp.values()]
    )

def check_valid_1(occupancy, day, lower_occupancy=125, upper_occupancy=300):
    return occupancy[day] >= lower_occupancy and occupancy[day] <= upper_occupancy

def check_valid_1_low(occupancy, day, lower_occupancy=125):
    return occupancy[day] >= lower_occupancy

def check_valid_1_up(occupancy, day, upper_occupancy=300):
    return occupancy[day] <= upper_occupancy

def find_invalid_low(occupancy, lower_occupancy=125):
    return [(d, occupancy[d]) for d in range(1, N_days+1) if occupancy[d] < lower_occupancy]

def find_invalid_up(occupancy, upper_occupancy=300):
    return [(d, occupancy[d]) for d in range(1, N_days+1) if occupancy[d] > upper_occupancy]

def deep_check(assigned_day, family_on_day, occupancy, lower_occupancy=125, upper_occupancy=300):
    # check assigned_day on family_on_day
    for ifamily, day in enumerate(assigned_day):
        assert ifamily in family_on_day[day], 'family %d assigned to %d day but not on family_on_day' % (ifamily, day)
    
    # check family_on_day on assigned_day
    for day, families in enumerate(family_on_day):
        for ifamily in families:
            assert assigned_day[ifamily] == day, 'family_on_day: %d %d, assigned_day: %d' % (ifamily, day, assigned_day[ifamily])

    # check occupancy
    for day, families in enumerate(family_on_day):
        if day == 0:
            continue
        count = sum([n_people[ifamily] for ifamily in families])
        assert count == occupancy[day], 'occupancy on day %d inconsist: (count, occupancy) %d %d' % (day, count, occupancy[day])

    # check valid
    assert check_valid_all(occupancy, lower_occupancy=lower_occupancy, upper_occupancy=upper_occupancy), 'occupancy out of bound'

    print('deep check: everything looks fine.')
    return True

