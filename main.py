from pynlpdoe.anova import ANOVA
import numpy as np

list_of_seeds = [5, 42] # repetitions
list_of_models = ['bert-base-cased', 'dslim/bert-base-NER']
#['tner/wikineural', 'pt'] ## Esse dataset tem um problema com labels/ids
list_of_datasets = [['wnut_17'], ['conll2003']] 

factors = ['Model Size', 'NER dataset']
levels_list = [list_of_models, list_of_datasets]
replicates = len(list_of_seeds)
total = np.prod([len(x) for x in levels_list]) * replicates
measures = np.random.rand(total) # type: ignore #([28,25,27,18,19,23,36,32,32,31,30,29,28,25,22,18,19,23,12,32,40,31,30,29]
anova = ANOVA(factors=factors, levels_list=levels_list, measures=measures, replicates=replicates)
print(anova.exec_anova())
#anova = ANOVA()
#print(anova.dummy_test())

