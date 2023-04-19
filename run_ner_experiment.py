from pynlpdoe.experiments import Treatment, Dataset
from pynlpdoe.anova import ANOVA
from itertools import product
import os
os.environ['TRANSFORMERS_CACHE'] = '/work/u4vn/cache'

list_of_seeds = [5, 42] # repetitions
list_of_models = ['bert-base-cased', 'bert-large-cased']
list_of_datasets = [['tner/wikineural', 'pt'], ['conll2003']]

factors = ['Model Size', 'NER dataset']
levels_list = [list_of_models, list_of_datasets]
replicates = len(list_of_seeds)

experiment = product(*levels_list)
results = []
f1 = []
for model_name, dataset_name in experiment:
	ds = Dataset(dataset_name)
	for s in list_of_seeds:
		print('============ REALIZANDO EXPERIMENTO ==============')
		print(model_name,dataset_name, s)
		treatment = Treatment(dataset=ds, model_name=model_name, seed=s)

		result = treatment.exec_treatment()
		results.append(result)
		f1.append(result['f1']) # type: ignore

print(results)
print(f1)

anova = ANOVA(factors=factors, levels_list=levels_list, measures=f1, replicates=replicates)
print(anova.exec_anova())

