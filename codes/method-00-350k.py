from collections import defaultdict
import numpy as np
import pandas as pd


###############
## Functions ##
###############

# the penalty term for Nd outside [125, 300]
penalty_oob = (300.-125.)/400.*300.**(.5+abs(300.-0.)/50.)

def delta_penalty(Nd_arr, n_people):

    '''
    The function for calculating the change in the penalty term for putting
    n_people to each day.

    Inputs
    ----
    Nd_arr: np.array.  The distribution of Nd (before adding in n_people).

    n_people: the number of people to put in.


    Output
    ----
    delta_penalty_final: an array with a size of 100, for each of the 100 days.
    '''

    # the original penalty for Nd_arr
    penalty_0 = (Nd_arr-125.)/400.*Nd_arr**(.5+abs(np.concatenate([np.diff(Nd_arr), [0.]]))/50.)
    mask_oob_below = (Nd_arr < 125.)
    mask_oob_above = (Nd_arr > 300.)
    ## penalty for Nd terms outside [125, 300]
    ## Linear function, with large jumps at 125 and 300 and slopes tilting
    ## inwards (from small numbers to 125 & from large numbers to 300).
    ## The slopes are implemented to encourge movements from outside
    ## towards the boundaries.
    penalty_0[mask_oob_below] = penalty_oob*(125.-Nd_arr[mask_oob_below])
    penalty_0[mask_oob_above] = penalty_oob*(Nd_arr[mask_oob_above]-300.)
    penalty_0 = np.sum(penalty_0)

    # the penalty if adding n_people to each of the 100 days
    base_arr = np.tile(Nd_arr, (100, 1)) + np.eye(100)*n_people
    penalty_base = (base_arr-125.)/400.\
                   *base_arr**(.5+abs(np.concatenate([np.diff(base_arr, axis=1), np.zeros((100, 1))], axis=1))/50.)
    ## penalty for Nd terms outside [125, 300]
    mask_oob_below = (base_arr < 125.)
    mask_oob_above = (base_arr > 300.)
    penalty_base[mask_oob_below] = penalty_oob*(125.-base_arr[mask_oob_below])
    penalty_base[mask_oob_above] = penalty_oob*(base_arr[mask_oob_above]-300.)

    # the change in penalty for adding n_people to each of the 100 days
    delta_penalty_final = np.sum(penalty_base, axis = 1)-penalty_0

    return delta_penalty_final

def cost_choices(n_people):

    '''
    Quick lookup.
    '''

    return np.array([0., 50., 50.+9.*n_people, 100.+9.*n_people, 200.+9.*n_people,\
                     200.+18.*n_people, 300.+18.*n_people, 300.+36.*n_people, 400.+36.*n_people, 500.+235.*n_people])

def compensation(choices_i, n_people):

    '''
    Calculate the preference cost term for a family of n_people for each day.

    Inputs
    ----
    choices_i: integer array.  The top ten choices of the family.

    n_people: integer.  Number of people in the family.
    '''

    # compensation for adding n_people to each of the 100 days
    compensation_final = np.ones(100)*(500.+434.*n_people)
    ## Note that the days are numbered from 1 to 100 (instead of starting at 0)
    compensation_final[(choices_i-1)] = cost_choices(n_people)

    return compensation_final


##########
## Data ##
##########
choices = pd.read_csv('../input/family_data.csv')
answer = pd.read_csv('../input/sample_submission.csv')

#################
## Calculation ##
#################

#### STEP 1. ####
# 1. create a distribution that satistifes the constraints by filling each
#    family in a random order to the position that minimally increases the
#    total cost.

# Create a table for the answer, based on the example.
answer_temp = answer.copy()
answer_temp['assigned_day'] = np.nan
answer_temp['final_choice'] = np.nan
answer_temp['n_people'] = choices['n_people']

# Shuffle the order of the family.
draws = np.random.permutation(choices['family_id'].values)

# Create a container for the Nd distribution (as a function of day).
Nd_accumulate = np.zeros(100)

# Loop through all families.
for i in range(len(draws)):

    # Input: family_id, choices, n_people
    draw_i = draws[i]
    choices_i = choices.values[draw_i, 1:-1]
    n_people_i = choices['n_people'].values[draw_i]

    # Use the function above to calculate the change in the total cost, as a
    # function of day.
    delta_cost = delta_penalty(Nd_accumulate, n_people_i)+compensation(choices_i, n_people_i)

    # Find the day that minimizes the change in the total cost.
    assigned_day_i = np.arange(1., 101.)[np.argmin(delta_cost)]
    ## Update the table.
    answer_temp.at[draw_i, 'assigned_day'] = assigned_day_i
    answer_temp.at[draw_i, 'final_choice'] = np.sum(np.arange(10.)*np.equal(choices_i, assigned_day_i))\
                                             +(-1.)*(~np.sum(np.equal(choices_i, assigned_day_i), dtype = bool))
    ## Update the distribution.
    Nd_accumulate[np.argmin(delta_cost)] += n_people_i


#### STEP 2. ####
# 2. Shaking: random draw a fmily, calculate the change in the total cost for
#    moving the family to each of the 100 days.  Then, move the family to the
#    day that lowers the cost.

# Input: number of shakes.
n_shakes = 100000

# Document the cost history.
cost_history = defaultdict(list)

# Quick lookup table for the preference cost.
comp0 = np.tile(np.array([0., 50., 50., 100., 200., 200., 300., 300., 400., 500., 500.]), (len(choices), 1))
comp1 = np.tile(np.array([0., 0., 9., 9., 9., 18., 18., 36., 36., 235., 434.]), (len(choices), 1))
comp1 = (comp1.T*choices['n_people'].values).T
comp_lookup = comp0+comp1

# Shake it!!
for k in range(n_shakes):

    # Input: family_id, choices, n_people
    draw_k = np.random.choice(choices['family_id'].values, 1)[0]
    choices_k = choices.values[draw_k, 1:-1]
    n_people_k = choices['n_people'].values[draw_k]

    # Calculate the Nd distribution based on the table.  The distribution is
    # needed as an input for the functions.
    Nd_accumulate = answer_temp.groupby('assigned_day')['n_people'].sum().values
    ## This is to take the selected family (family_id == draw_k) out from the
    ## Nd distribution.
    Nd_accumulate[int(answer_temp['assigned_day'].values[draw_k])-1] -= n_people_k

    # Calculate the change in cost for moving the family to each day.
    delta_cost = delta_penalty(Nd_accumulate, n_people_k)+compensation(choices_k, n_people_k)

    # Assign a new day for the family, based on the change in cost.
    assigned_day_k = np.arange(1., 101.)[np.argmin(delta_cost)]
    ## Update the table.
    answer_temp.at[draw_k, 'assigned_day'] = assigned_day_k
    answer_temp.at[draw_k, 'final_choice'] = np.sum(np.arange(10.)*np.equal(choices_k, assigned_day_k))\
                                             +(-1.)*(~np.sum(np.equal(choices_k, assigned_day_k), dtype = bool))
    ## Update the Nd distribution.
    Nd_accumulate[np.argmin(delta_cost)] += n_people_k


    # Record the cost
    ## The penalty term.
    penalty = (Nd_accumulate-125.)/400.*Nd_accumulate**(.5+abs(np.concatenate([np.diff(Nd_accumulate), [0.]]))/50.)
    penalty = np.sum(penalty)
    ## The preference cost term.
    compensation_total = 0.
    final_choices = answer_temp['final_choice'].values
    final_choices[final_choices == -1.] = 10.
    compensation_total = np.sum(np.choose(final_choices.astype(int), comp_lookup.T))

    ## Update the cost_history table.
    cost_history['penalty'].append(penalty)
    cost_history['compensation_total'].append(compensation_total)
    cost_history['cost_total'].append(penalty+compensation_total)

# Yay for pandas!!
cost_history = pd.DataFrame(cost_history)
print('----Final Score----')
print('Accounting Penalty:', cost_history['penalty'].values[-1])
print('Preference Cost:', cost_history['compensation_total'].values[-1])
print('Score:', cost_history['cost_total'].values[-1])
print('-------------------')

#####################
## Save the result ##
#####################

## Note that answer_temp has information of the n_people and the final choice
## assigned to the family; this is different from the requirement for submission.
answer_temp.to_csv('../output/answer_temp.csv', index = False)
cost_history.to_csv('../output/cost_history.csv', index = False)
