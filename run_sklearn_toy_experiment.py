from pynlpdoe.experiments import Treatment, Dataset
from pynlpdoe.anova import ANOVA
from itertools import product
import os
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split

os.environ['TRANSFORMERS_CACHE'] = '/work/u4vn/cache'


replicates = 10
list_of_seeds = np.random.randint(100, size=replicates)  #[5, 42, 33, 22, 49] # repetitions

factors = ['Num Hidden Units', 'Solver']
levels_list = [[4, 64],['adam', 'sgd']]

# df = pd.read_csv('./sample_data/train.csv')
# y = df['label'].values
# X = df.drop('label', axis=1).values

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=list_of_seeds[0])

experiment = product(*levels_list)
measures = []

for p1, p2 in experiment:
    
	for s in list_of_seeds:
		
		np.random.seed(s)
		print('============ Experiment ==============')
		print(list(zip(factors, (p1,p2))), 'seed', s)
		#model = RandomForestClassifier(max_depth=p1, n_estimators=p2, random_state=s)
		#current_treatment = [p1 for i in range(p2)]
		model = MLPClassifier(hidden_layer_sizes=(p1,), 
                       random_state=s, activation='logistic', 
                       shuffle=True, solver=p2, max_iter=100)
		model.fit(X_train, y_train)
		score = model.score(X_test,y_test)
		print(score)
		measures.append(score)

print(measures)

anova = ANOVA(factors=factors, levels_list=levels_list, measures=measures, replicates=replicates)
df = anova.generate_dataframe()
df.to_csv('./experiment_data.csv', index=False)
analysis = anova.exec_anova()
print(analysis)
anova.test_assumptions()
