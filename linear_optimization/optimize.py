from mip import *
import pandas as pd 
import numpy as np 

# Import dataset
df = pd.read_csv('data/property.csv')

# Length 
nb_ward = len(df.ward)

# Declare the model 
m = Model("Resource-Allocation", solver_name="CBC")

# Declare the variables for assignment of students  
x = [m.add_var(var_type='I') for i in range(nb_ward)]

# number hours in total a month
nb_hour = 100 * 2 * 4 * 4 
print(f'Total number of hours per month for the whole Barnet: {nb_hour}')
# Maximum hours per ward
max_hour = (22 - 8) * 4 * 4
print(f'Maximum number of hours can assign to a single ward: {max_hour}')
# Min hours per ward
min_hour = 0 * 4 * 4 
print(f'Min number of hours can assign to a single ward: {min_hour}')

# Total hours is smaller than nb_hour 
m += xsum(w for w in x) <= nb_hour 

# Hours allocated to each ward in [min_hours, max_hours]
for i in range(nb_ward):
    m += x[i] <= max_hour
    m += x[i] >= min_hour

# Remove extremity 
q_1 = (max_hour - min_hour)/4
q_2 = (max_hour - min_hour)/2
q_3 = (max_hour - min_hour)*(3/4)
for key in ['density', 'camera', 'imd_value']:
    m += xsum(x[i]*df[key].iloc[i] for i in range(nb_ward) if x[i] < q_2)

# Set the weight 
weight = {
    'density' : 0,
    'predicted_crime' : 1,
    'camera' : 0,
    'imd_value' : 0, 
}

# Objective function 
m.objective = maximize(
    xsum(
        - df['density'].iloc[i] * x[i] * weight['density'] 
        + df['predicted_crime'].iloc[i] * x[i] * weight['predicted_crime']
        - df['camera'].iloc[i] * x[i] * weight['camera'] 
        + df['imd_value'].iloc[i] * x[i] * weight['imd_value']
    for i in range(nb_ward)
    )
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
    print(f'Assign to {df.ward[i]} in total: {x[i].x} hours')

print(f'Used in total: {sum([w.x for w in x])} hours')