from mip import *
import pandas as pd 
import numpy as np 

# Import dataset
df = pd.read_csv('data/property.csv')

# Length 
nb_ward = len(df.ward)
nb_f = 4
# Declare the model 
m = Model("Resource-Allocation", solver_name="CBC")

# Declare the variables for assignment of students  
x = [[m.add_var(var_type='I') for i in range(nb_ward)] for f in range(nb_f)]

# number hours in total a month
nb_hour = 100 * 2 * 4 * 4 
print(f'Total number of hours per month for the whole Barnet: {nb_hour}')
# Maximum hours per ward
max_hour = (22 - 8) * 4 * 4
print(f'Maximum number of hours can assign to a single ward: {max_hour}')
# Min hours per ward
min_hour = 4 * 4 * 4 
print(f'Min number of hours can assign to a single ward: {min_hour}')

# Total hours is smaller than nb_hour 
m += xsum(sum(f[i] for f in x) for i in range(nb_ward)) <= nb_hour 

# Hours allocated to each ward in [min_hours, max_hours]
for i in range(nb_ward):
    m += xsum(x[f][i] for f in range(nb_f)) <= max_hour
    m += xsum(x[f][i] for f in range(nb_f)) >= min_hour

# Set the weight 
weight = {
    'density' : 0.2,
    'predicted_crime' : 0.3,
    'camera' : 0.3,
    'imd_value' : 0.2, 
}

# Objective function 
m.objective = maximize(
    - sum(df['density'].iloc[i] * x[0][i] * weight['density'] for i in range(nb_ward))
    + sum(df['predicted_crime'].iloc[i] * x[1][i] * weight['predicted_crime'] for i in range(nb_ward))
    - sum(df['camera'].iloc[i] * x[2][i] * weight['camera'] for i in range(nb_ward))
    + sum(df['imd_value'].iloc[i] * x[3][i] * weight['imd_value'] for i in range(nb_ward))
)

# Printing information about the model
print('model has {} vars, {} constraints and {} nzs'.format(m.num_cols, m.num_rows, m.num_nz))

# Allowing logs
m.store_search_progress_log = True

# Optimize
m.optimize()

# Print logs
print("Logs")
for i in m.search_progress_log.log:
    print(round(i[0],3), "\t", round(i[1][0],3), "\t", round(i[1][1],3))

# Print output (if a solution is found)
if m.num_solutions:
    print("Solution of value",round(m.objective_value,2))

# Print out the solution 
for i in range(nb_ward):
    print(f'Assign to {df.ward[i]} in total: {sum(x[f][i].x for f in range(nb_f))} hours')

print(f'Used in total: {sum(sum(f[i].x for f in x) for i in range(nb_ward))} hours')