from sklearn.linear_model import LogisticRegression
from utils import genre_classification

class logistic_regression(genre_classification):
    def __init__(self):
        print("initializing logistic regression classifier")
        
    def gen(self, n_estimators=100, random_state=42):
        self.model = LogisticRegression(max_iter=500)
        return self.model