import pandas as pd
import pingouin as pg
import numpy as np
from itertools import product


class ANOVA:

	def __init__(self, measures, factors, levels_list, replicates):
		# factors = ['FatorA' ,'FatorB', 'FatorC']
		# levels_list = [['low','high'],['L','H'],['-','+','++']]
		# replicates = 5
     
		self.measures = measures
		self.factors = factors
		self.levels_list = levels_list
		self.replicates = replicates
	 
	def generate_dataframe(self):
	
		lines = []
		for factor_combination in product(*self.levels_list):
			line = {}
			for idx, factor in enumerate(self.factors):
				line[factor] = str(factor_combination[idx])
			for k in range(self.replicates):
				lines.append(line)
		df = pd.DataFrame(lines,columns=self.factors)
		df['y'] = self.measures
		return df

	def exec_anova(self):

		df = self.generate_dataframe() 

		return df.anova(dv='y', between=self.factors, ss_type=3).round(3) # type: ignore