import numpy as np
from scipy.stats import f

def get_coefficient_matrix(levels_list, replicates):
    num_factors = len(levels_list)
    num_treatments = np.prod(levels_list)
    coefficient_matrix = np.zeros((replicates*num_treatments, num_treatments))

    for i in range(num_treatments):
        for j in range(replicates):
            coefficient_matrix[i*replicates+j][i] = 1
    
    for i in range(num_treatments):
        for j in range(num_factors):
            factor_level = int(i / np.prod(levels_list[j+1:]))
            coefficient_matrix[i][i - factor_level*np.prod(levels_list[j+1:])] = (factor_level - (levels_list[j]-1)/2)

    return coefficient_matrix



def calculate_ANOVA_metrics(data, levels_list, replicates):
    num_factors = len(levels_list)
    num_treatments = np.prod(levels_list)
    grand_mean = np.mean(data)
    ss_total = np.sum((data - grand_mean)**2)
    ss_factor = 0
    ss_error = 0

    coefficient_matrix = get_coefficient_matrix(levels_list, replicates)
    projection_matrix = np.dot(np.linalg.inv(np.dot(coefficient_matrix.T, coefficient_matrix)), coefficient_matrix.T)

    for i in range(num_treatments):
        factor_mean = np.mean(data[i*replicates:(i+1)*replicates])
        ss_factor += replicates*(factor_mean - grand_mean)**2
        ss_error += np.sum((data[i*replicates:(i+1)*replicates] - factor_mean)**2)

    dof_factor = num_factors - 1
    dof_error = num_treatments*replicates - num_factors
    dof_total = num_treatments*replicates - 1

    ms_factor = ss_factor/dof_factor
    ms_error = ss_error/dof_error
    f_statistic = ms_factor/ms_error
    p_value = f.sf(f_statistic, dof_factor, dof_error)

    return {
        'Source of variation': ['Factor', 'Error', 'Total'],
        'Sum of squares': [ss_factor, ss_error, ss_total],
        'Degrees of freedom': [dof_factor, dof_error, dof_total],
        'Mean square': [ms_factor, ms_error, '-'],
        'F statistic': [f_statistic, '-', '-'],
        'p value': [p_value, '-', '-']
    }


y_measures = np.array([28,25,27,36,32,32,18,19,23,31,30,29])
levels_list = [2,2]
replicates = 3

calculate_ANOVA_metrics(y_measures, levels_list, replicates)
