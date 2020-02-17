import numpy as np
import pandas as pd

N_families = 5000
N_days = 100
penalty_map_size = 1600

fixed_family_cost = np.array([0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500])
fixed_person_cost = np.array([0, 0, 9, 9, 9, 18, 18, 36, 36, 36+199, 36+398])

# array of preference cost per family
# preference_cost_per_family[n_people, i_choice]
preference_cost_per_family = (
    fixed_family_cost.reshape(1, -1) +
    np.arange(0, 9).reshape(-1, 1) * fixed_person_cost.reshape(1, -1)
)

# ============ Read family data ============
family_data = pd.read_csv('../input/family_data.csv')
n_people = family_data['n_people'].values
choices = family_data.loc[:, ['choice_'+str(i) for i in range(10)]].values

# table from (family_id, choice) to preference cost
family_id_choice_to_pref_cost = (
    fixed_family_cost.reshape(1, -1) + n_people.reshape(-1, 1) * fixed_person_cost.reshape(1, -1)
)

# table from (family_id, day) to preference cost
family_id_days_to_pref_cost = np.repeat(family_id_choice_to_pref_cost[:, [-1]], N_days + 1, axis=1)
for f in range(len(family_id_days_to_pref_cost)):
    family_id_days_to_pref_cost[f, choices[f]] = family_id_choice_to_pref_cost[f, :-1]
# the zero day is not legal, put a huge number here to avoid bug
family_id_days_to_pref_cost[:, 0] = 99999

# ============ Total Cost ============
def cal_total_preference(assigned_day):
    """ Calculate total preference cost of an assigned_day arangement """
    return np.sum(family_id_days_to_pref_cost[np.arange(N_families), assigned_day])

def cal_total_accounting(occupancy):
    """ Calculate total accounting penality of an occupancy arangement """
    assert len(occupancy) == N_days + 2
    occupancy = occupancy[1:-1]
    return np.sum(
        (occupancy - 125) / 400 *
        occupancy ** (
            0.5 + np.abs(np.concatenate([np.diff(occupancy), [0]])) / 50
        )
    )

def cal_total(assigned_day, occupancy):
    """ Calculate total score (cost) of a configuration of assigned_day and occupancy """
    return cal_total_preference(assigned_day) + cal_total_accounting(occupancy)

# ============ Difference of Exchange Cost ============
# precomputed accounting penalty
# table from [nd, nd+1] to accounting penalty of the d day
# where nd is the occupancy of the d day
nd_ndp1_to_account_penality = np.array([
    [
        (nd - 125) / 400 * nd ** (0.5 + abs(nd - ndp1) / 50)
        for ndp1 in range(penalty_map_size + 1)
    ] 
    for nd in range(penalty_map_size + 1)
])

# Move 1 family
def cal_diff_1_accounting(occupancy, day0, day1, n):
    """ Difference in the accounting penalty for moving a family of size n from day0 to day1 
        given the current occupancy at day0.
        Note: day0, day1, or n don't support vertorization
    """
    # set of days that might affect their accounting penalties
    day_set = set([day0 - 1, day0, day1 - 1, day1])

    diff = 0.

    for day in day_set:
        if day == 100:
            diff -= nd_ndp1_to_account_penality[occupancy[day], occupancy[day]]
        elif day != 0:
            diff -= nd_ndp1_to_account_penality[occupancy[day], occupancy[day + 1]]

    occupancy[day0] -= n
    occupancy[day1] += n

    for day in day_set:
        if day == 100:
            diff += nd_ndp1_to_account_penality[occupancy[day], occupancy[day]]
        elif day != 0:
            diff += nd_ndp1_to_account_penality[occupancy[day], occupancy[day + 1]]

    occupancy[day0] += n
    occupancy[day1] -= n
    return diff

def cal_diff_1(family_id, occupancy, day0, day1):
    """ Difference in the toal cost for moving the family family_id from day0 to day1 
        given the current occupancy at day0.
        Note: family_id, day0, or day1 don't support vertorization
    """
    return (
        family_id_days_to_pref_cost[family_id, day1] - family_id_days_to_pref_cost[family_id, day0]
        + cal_diff_1_accounting(occupancy, day0, day1, n_people[family_id])
    )

# Vectorized version
def cal_diff_1_accounting_vec(occupancy, day0, day1, n):
    """ Difference in the accounting penalty for moving a family of size n from day0 to day1 
        given the current occupancy at day0. (only day1 support numpy array)
    """
    return np.array([cal_diff_1_accounting(occupancy, day0, d, n) for d in day1])
# def cal_diff_1_accounting_vec(occupancy, day0, day1, n):
#     """ Difference in the accounting penalty for moving a family of size n from day0 to day1 
#         given the current occupancy at day0. (vectorized version of cal_diff_1_accounting)
#         Note: day0, day1, or n support vertorization!
#     """
#     occ_day0m1 = occupancy[day0 - 1]
#     occ_day0 = occupancy[day0]
#     occ_day0p1 = occupancy[day0 + 1]
#     occ_day1m1 = occupancy[day1 - 1]
#     occ_day1 = occupancy[day1]
#     occ_day1p1 = occupancy[day1 + 1]

#     return (
#         # difference in day 0 accounting penalty
#         (
#             nd_ndp1_to_account_penality[occ_day0m1, occ_day0 - n] -
#             nd_ndp1_to_account_penality[occ_day0m1, occ_day0]
#         ) * (day0 != 1) +
#         np.where(
#             day0 != 100,
#             (
#                 nd_ndp1_to_account_penality[occ_day0 - n, occ_day0p1] -
#                 nd_ndp1_to_account_penality[occ_day0, occ_day0p1]
#             ),
#             (
#                 nd_ndp1_to_account_penality[occ_day0 - n, occ_day0 - n] -
#                 nd_ndp1_to_account_penality[occ_day0, occ_day0]
#             )
#         ) +
#         # difference in day 1 accounting penalty
#         (
#             nd_ndp1_to_account_penality[occ_day1m1, occ_day1 + n] -
#             nd_ndp1_to_account_penality[occ_day1m1, occ_day1]
#         ) * (day1 != 1) +
#         np.where(
#             day1 != 100,
#             (
#                 nd_ndp1_to_account_penality[occ_day1 + n, occ_day1p1] -
#                 nd_ndp1_to_account_penality[occ_day1, occ_day1p1]
#             ),
#             (
#                 nd_ndp1_to_account_penality[occ_day1 + n, occ_day1 + n] -
#                 nd_ndp1_to_account_penality[occ_day1, occ_day1]
#             )
#         )
#     )

def cal_diff_1_vec(family_id, occupancy, day0, day1):
    """ Difference in the toal cost for moving the family family_id from day0 to day1
        given the current occupancy at day0. (vectorized version)
        Note: only day1 support vertorization and has to be numpy array
    """
    return (
        family_id_days_to_pref_cost[family_id, day1] - family_id_days_to_pref_cost[family_id, day0]
        + cal_diff_1_accounting_vec(occupancy, day0, day1, n_people[family_id])
    )

# Move N families
def cal_diff_n_accounting(occupancy, day0s, day1s, family_sizes):
    """ Difference in the accounting penalty for moving a list of families
        (each with sizes family_sizes) from day0s to day1s correspondingly
        given the occupancy before move.
        occupancy [np.array]: occupancy array before move
        days0s [np.array]: day0 of each family
        days1s [np.array]: day1 of each family
        family_sizes [np.array]: family sizes of each family
    """
    # set of days that might affect their accounting penalties
    day_set = set(day0s - 1) | set(day0s) | set(day1s - 1) | set(day1s)
    
    diff = 0.
    
    for day in day_set:
        if day == 100:
            diff -= nd_ndp1_to_account_penality[occupancy[day], occupancy[day]]
        elif day != 0:
            diff -= nd_ndp1_to_account_penality[occupancy[day], occupancy[day+1]]
    
    for day0, day1, n in zip(day0s, day1s, family_sizes):
        occupancy[day0] -= n
        occupancy[day1] += n
    
    for day in day_set:
        if day == 100:
            diff += nd_ndp1_to_account_penality[occupancy[day], occupancy[day]]
        elif day != 0:
            diff += nd_ndp1_to_account_penality[occupancy[day], occupancy[day+1]]

    for day0, day1, n in zip(day0s, day1s, family_sizes):
        occupancy[day0] += n
        occupancy[day1] -= n
    return diff

def cal_diff_n(family_ids, occupancy, day0s, day1s, family_sizes):
    """ Difference in the total cost for moving family_ids
        (each with sizes family_sizes) from day0s to day1s correspondingly
        given the occupancy before move.
        Note: don't support vertorization
        family_ids [np.array or list]
        occupancy [np.array]: occupancy array before move
        days0s [np.array]: day0 of each family
        days1s [np.array]: day1 of each family
        family_sizes [np.array]: family sizes of each family
    """
    return (
        family_id_days_to_pref_cost[family_ids, day1s].sum()
        - family_id_days_to_pref_cost[family_ids, day0s].sum()
        + cal_diff_n_accounting(occupancy, day0s, day1s, family_sizes)
    )
