from pynlpdoe.experiments import Treatment, Dataset
from pynlpdoe.anova import ANOVA
from itertools import product
import os
from tqdm import tqdm

os.environ['TRANSFORMERS_CACHE'] = '/work/u4vn/cache'

list_of_seeds = [5, 42] # repetitions
list_of_models = ['bert-base-cased', 'dslim/bert-base-NER']
#['tner/wikineural', 'pt'] ## Esse dataset tem um problema com labels/ids
list_of_datasets = [['wnut_17'], ['conll2003']] 

factors = ['Model Size', 'NER dataset']
levels_list = [list_of_models, list_of_datasets]
replicates = len(list_of_seeds)

experiment = product(*levels_list)
results = []
f1 = []
for model_name, dataset_name in tqdm(experiment):
	ds = Dataset(dataset_name)
	for s in list_of_seeds:
		print('============ REALIZANDO EXPERIMENTO ==============')
		print(model_name,dataset_name, s)
		treatment = Treatment(dataset=ds, model_name=model_name, seed=s)

		result = treatment.exec_treatment()
		results.append(result)
		f1.append(result['test_f1']) # type: ignore

print(results)
print(f1)

anova = ANOVA(factors=factors, levels_list=levels_list, measures=f1, replicates=replicates)
print(anova.exec_anova())

# [0.0, 0.0, 0.82239916911892, 0.8169306673591276, 0.39388145315487566, 0.37324840764331213, 0.9037517531556802, 0.9037517531556802]
#                      Source     SS  DF     MS          F  p-unc    np2
# 0                Model Size  0.109   1  0.109   1919.975    0.0  0.998
# 1               NER dataset  0.898   1  0.898  15760.281    0.0  1.000
# 2  Model Size * NER dataset  0.045   1  0.045    787.373    0.0  0.995
# 3                  Residual  0.000   4  0.000        NaN    NaN    NaN