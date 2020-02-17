choice_0: no consolation gifts
choice_1: one $50 gift card to Santa's Gift Shop
choice_2: one $50 gift card, and 25% off Santa's Buffet (value $9) for each family member
choice_3: one $100 gift card, and 25% off Santa's Buffet (value $9) for each family member
choice_4: one $200 gift card, and 25% off Santa's Buffet (value $9) for each family member
choice_5: one $200 gift card, and 50% off Santa's Buffet (value $18) for each family member
choice_6: one $300 gift card, and 50% off Santa's Buffet (value $18) for each family member
choice_7: one $300 gift card, and free Santa's Buffet (value $36) for each family member
choice_8: one $400 gift card, and free Santa's Buffet (value $36) for each family member
choice_9: one $500 gift card, and free Santa's Buffet (value $36) for each family member, and 50% off North Pole Helicopter Ride tickets (value $199) for each family member
'
import numpy as np

# constants #
N_all_people = 21003
N_families = 5000
N_days = 100
# constants #

# params #
cutoff = 5
target_cost = 
# params #

# functions #
cost_per_family = [0,50,50,100,200,200,300,300,400,500]
cost_per_person = [0,0,9,9,9,18,18,36,36,235]

def cal_pref_cost(family_counts, people_counts, cutoff=cutoff):
    return np.dot(family_counts[:cutoff], cost_per_family[:cutoff]) + np.dot(peopl_counts[:cutoff], cost_per_person[:cutoff])
# functions #

# init
family_counts = []
people_counts = []

assert sum(family_counts) == N_families, 'family number incorrect: %d, %d.' % (sum(family_counts), N_families)
assert sum(people_counts) == N_all_people, 'people number incorrect: %d, %d' % (sum(people_counts), N_all_people)
