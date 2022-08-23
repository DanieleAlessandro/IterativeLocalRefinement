from Formula import Predicate
from Godel import *
# from Product import *

#  ===== KNOWLEDGE =====

predicates = {}
# Generate predicates
for x in range(10):
    name_x = 'x_' + str(x)
    predicates[name_x] = Predicate(name_x, x)
    name_y = 'y_' + str(x)
    predicates[name_y] = Predicate(name_y, x + 10)

# Generate rules
knowledge = []
for x in range(10):
    for y in range(10):
        name_x = 'x_' + str(x)
        name_y = 'y_' + str(y)

        s = x + y
        name_sum = 'sum_' + str(s)
        if name_sum not in predicates.keys():
            predicates[name_sum] = Predicate(name_sum, len(predicates))

        knowledge.append(IMPLIES([AND([predicates[name_x], predicates[name_y]]), predicates[name_sum]]))
