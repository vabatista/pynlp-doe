from pynlpdoe.experiments import Model
from pynlpdoe.anova import ANOVA
m = Model()

anova = ANOVA()
print(anova.dummy_test())

