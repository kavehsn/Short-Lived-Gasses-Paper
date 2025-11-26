import os
import numpy as np

# -------------------------------------------
# Define the grid of \widetilde{T} values
# These will be inserted into the PBS template
# and each value corresponds to a separate job.
# -------------------------------------------
T_tilde_values = np.arange(26, 100, 5)

# -------------------------------------------
# Read the PBS job template.
# The template must contain the placeholder:
#     ${T_TILDE}
# which will be replaced for each job.
# -------------------------------------------
with open('Optimization_SGD_PBS.pbs', 'r') as file:
    job_template = file.read()

# -------------------------------------------
# Loop over all \widetilde{T} values, create
# a dedicated job file, and submit it via qsub.
# -------------------------------------------
for T_tilde in T_tilde_values:
    
    # Replace placeholder with actual value
    job_script = job_template.replace('${T_TILDE}', str(T_tilde))
    
    # Name of the PBS job file for this run
    job_filename = f'job_{T_tilde}.pbs'
    
    # Write the expanded job script
    with open(job_filename, 'w') as file:
        file.write(job_script)
    
    # Submit the job to the scheduler
    os.system(f'qsub {job_filename}')
