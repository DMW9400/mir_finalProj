from sklearn.linear_model import LogisticRegression
from utils import genre_classification

class logistic_regression(genre_classification):
    def __init__(self):
        print("initializing logistic regression classifier")
        
    def gen(self, random_state=42, solver = 'lbfgs'):
        self.model = LogisticRegression(solver=solver, max_iter=10000, random_state=random_state)
        return self.model