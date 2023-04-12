import pandas as pd
import pingouin as pg
import numpy as np
from itertools import product


class ANOVA:

	def generate_dataframe(self, measures, factors, levels_list, replicates):
    
		lines = []
		for factor_combination in product(*levels_list):
			line = {}
			for idx, factor in enumerate(factors):
				line[factor] = factor_combination[idx]
			for k in range(replicates):
				lines.append(line)
		df = pd.DataFrame(lines,columns=factors)
		df['y'] = measures
		#df['trial'] = list(range(replicates)) * int(len(measures) / replicates)
		return df

	def dummy_test(self):

		factors = ['FatorA' ,'FatorB', 'FatorC']
		levels_list = [['low','high'],['L','H'],['-','+','++']]
		replicates = 5
		total = np.prod([len(x) for x in levels_list]) * replicates
		y_measures = np.random.rand(total) #([28,25,27,18,19,23,36,32,32,31,30,29,28,25,22,18,19,23,12,32,40,31,30,29])
		df = self.generate_dataframe(y_measures, factors, levels_list, replicates)

		return df.anova(dv='y', between=factors, ss_type=3).round(3)