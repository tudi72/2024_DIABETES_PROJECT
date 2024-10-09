from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin,BaseEstimator
from pypots.imputation import SAITS

class ImputerTrainer(BaseEstimator,TransformerMixin):
    def __init__(self,models,param_grids,scoring_method = 'neg_mean_squared_error'):
        self.models = models
        self.param_grids = param_grids
        self.scoring_method = 'neg_mean_squared_error'
        self.best_models = {}
        self.X = None 
        self.y = None 

    def fit(self,X,y):
        self.X = X 
        self.y = y 
        return self 
    
    def transform(self,X):
        for model_name, model in self.models.items():
            param_grid = self.param_grids[model_name]

            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring=self.scoring_method,
                verbose=False,
                return_train_score=True,
            )

            grid_search.fit(self.X, self.y)
            cv_results = grid_search.cv_results_

            self.best_models[model_name] = grid_search.best_estimator_

            print("=" * 170)
            for i, params in enumerate(cv_results['params']):
                score = abs(cv_results['mean_test_score'][i])
                print(f"[{model_name}]:\t{score}\t\t{params}\t")
                
    def _info(self, msg:str) -> None:
        """Info print utility to know which class prints to terminal."""
        print("\033[36m[TrainerImputer]: ", msg, "\033[0m\n")



