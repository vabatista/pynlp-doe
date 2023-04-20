from pynlpdoe.anova import ANOVA

import numpy as np
import pandas as pd

y_measures = np.random.rand(2**3*3) #np.array([28,25,27,18,19,23,36,32,32,31,30,29,28,25,22,18,19,23,12,32,40,31,30,29])
factors = ["Facotr A" ,"Factor B", "Factor C"]
levels_list = [['low','high'],['low','high'],['low','high']]
replicates = 3

anova = ANOVA(factors=factors, levels_list=levels_list, measures=y_measures, replicates=replicates)
print(anova.exec_anova())
anova.test_assumptions()


