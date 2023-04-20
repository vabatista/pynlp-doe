import pandas as pd
import pingouin as pg
import numpy as np
from itertools import product
from scipy import stats

## TODO: Testar hipóteses de uso de ANOVA
class ANOVA:

	model = None
	df = None
 
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

		self.df = self.generate_dataframe() 

		self.model =  self.df.anova(dv='y', between=self.factors, ss_type=3).round(3) # type: ignore
		k = np.prod([len(i) for i in self.levels_list])  # Number of groups
		achieved_power = pg.power_anova(eta_squared=self.model.loc[0, 'np2'], k=k, n=self.replicates, alpha=0.05)
		print('Achieved power: %.4f' % achieved_power)

		return self.model

	def test_assumptions(self):
		#for factor in self.factors:
		if self.df is not None:
			
   			#for factor_combination in product([*self.factors]):
			samples = []
			for name, group in self.df.groupby(self.factors):
				samples = group['y'].values			
				k2, p = stats.normaltest(samples) # type: ignore
				print('Teste de normalidade para', name, p >= 0.05)
				
			k2, p = stats.levene(*samples) # type: ignore
			print('Teste de mesma variância entre os grupos', p >= 0.05)
	
				#print(pg.normality(self.df, group=self.factors, dv='y'))
				#print(pg.homoscedasticity(self.df, group=self.factors, dv='y'))
		
			