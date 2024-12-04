import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from utils import genre_classification


class random_forest(genre_classification):
    def __init__(self):
        print("initializing random forest classifier")
    
    def gen(self, n_estimators=1000, random_state=0):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        return self.model