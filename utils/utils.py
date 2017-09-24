import numpy as np
import pandas as pd


class OOFStacking():
    
    '''
    Out-of-fold stacking of scikit-learn models

    -----------------------------------------
    Parameters:
    
    meta_model : meta-estimator for stacking, scikit-learn estimator

    models : dict, in format {'model_name': [model, X, X_test, n_iterations]}.
        model_name - string name of model
        model - scikit-learn estimator
        X - array-like, shape [n_samples, n_features] - train data for model
        X_test - array-like, shape [n_samples, n_features] - test data for model
        n_itrations - int, (default=1)
            number of fit iterations with different random seeds for each model

    n_splits : int, (default=10)
        number of folds in out-of-fold.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    -----------------------------------------
    Attributes:

    df_stack_ : pd.DataFrame with shape (n_samples train, n_models)
        columns - predictions of each model on train data
        rows - train samples

    df_stack_test_ : pd.DataFrame with shape (n_samples test, n_models)
        columns - predictions of each model on test data
        rows - test samples

    meta_model_ : estimator, fitted meta-model

    '''
        
    
    def __init__(self, meta_model, models, n_splits=10, random_state=None):
        
        self.meta_model = meta_model
        self.models = models
        self.n_splits = n_splits
        self.random_state = random_state
    
    def fit(self, X, y, n_iterations=5):
        '''
        Out-of-fold prediction on train data (X, y)
        
        -----------------------------------------
        Parameters:
        
        X : array-like, shape [n_samples, n_features]
            The training input samples, if no special data for this model
            in models dict.
            
        y : array-like, shape [n_samples]
            The train target values.
            
        n_iterations : int, (default=5)
            Number of fit iterations with different random seeds
            for each model. Final prediction of model is average of
            all iterations for this model.
            
        -----------------------------------------
        Returns:
        
        self : object
            Returns self.
        '''
        
        try:
            X = X.values
        except:
            pass
        
        from sklearn.model_selection import KFold, train_test_split
        np.random.seed(self.random_state)
        kfold = KFold(n_splits=self.n_splits)
        self.X = X
        self.y = y
        self.n_iterations=n_iterations
        self.df_stack_ = pd.DataFrame(data=np.zeros((X.shape[0], len(self.models)), dtype=np.float64), 
                                        columns=self.models.keys())
        
        for est in self.models:
            
            print est
            try:
                # if pd.DataFrame
                X = self.models[est][1].values
                self.models[est][1] = X
            except:
                # if np.array
                X = self.models[est][1]
            
            try:
                n_iter = self.models[est][3]
            except:
                n_iter = self.n_iterations
            
            model = self.models[est][0]
            
            for train_index, test_index in kfold.split(X, y, y):
                
                for it in xrange(n_iter):                    
                    try:
                        model.random_state = 841*it + 4521  ## MORE MAGIC NUMBERS!!
                    except:
                        model.seed = 841*it + 4521
                    model = model.fit(X[train_index, :], y[train_index])
                    try:
                        self.df_stack_.loc[test_index, est] += model.predict_proba(X[test_index, :])[:,1]
                    except:
                        self.df_stack_.loc[test_index, est] += model.predict(X[test_index, :])
                
            self.df_stack_.loc[:, est] /= n_iter
                                
        return self
    
    def predict(self, X_test):
        '''
        Predict target values for X_test.
        
        -----------------------------------------
        Parameters:
        
        X_test : array-like, shape = [n_samples, n_features]
        
        -----------------------------------------
        Returns:
        
        y : array of shape = [n_samples]
            The predicted values.
        '''
        
        self.df_stack_test_ = pd.DataFrame(data=np.zeros((X_test.shape[0], len(self.models)), 
                                                        dtype=np.float64), columns=self.models.keys())
        
        for est in self.models:
        
            print est
            try:
                X_test = self.models[est][2].values
                self.models[est][2] = X_test
            except:
                X_test = self.models[est][2]

            try:
                n_iter = self.models[est][3]
            except:
                n_iter = self.n_iterations
        
            model = self.models[est][0]
        
            #predict for test data
            for it in xrange(n_iter):
                try:
                    model.random_state = 841*it + 4521
                except:
                    model.seed = 841*it + 4521
                model = model.fit(self.models[est][1], y)
                try:
                    self.df_stack_test_.loc[:, est] += model.predict_proba(X_test)[:,1]
                except:
                    self.df_stack_test_.loc[:, est] += model.predict(X_test)
          
        
            self.df_stack_test_.loc[:,est] /= n_iter
            
        scores = np.zeros((X_test.shape[0],), dtype=np.float64)
        
        #meta-model
        for it in xrange(self.n_iterations):
            try:
                self.meta_model.random_state = 841*it + 4521
            except:
                self.meta_model.seed = 841*it + 4521
            model = self.meta_model.fit(self.df_stack_, self.y)
            scores += model.predict(self.df_stack_test_)
        
        scores /= self.n_iterations
        return scores



def genetic_algorithm(estimator, X, y, scoring, cv=5, epochs=50, p0=0.5, p_cross=0.5, p_mut=0.01, N_best=50):
    """
    Implements genetic algorithm for feature selection

    -----------------------------------------
    Parameters:
    
    estimator : sklearn estimator
        
    X : pd.DataFrame, shape [n_samples, n_all_features]
        The train data with all features.
        
    y : array-like, shape
        Number of fit iterations with different random seeds
        for each model. Final prediction of model is average of
        all iterations for this model.

    scoring : str
        Type of scikit-learn scoring in cross-validation

    cv : int, defaul 5
        Number of folds

    epochs : int, default 50
        Number of epochs

    p0 : float, default 0.5
        Probability of including feature to sample 

    p_cross : float, default 0.5
        Probability of crossingover

    p_mut : float, default 0.5
        Probability of mutation

    N_best : int, default 50
        Number of best samples, selected for next epoch 
        
    -----------------------------------------
    Returns:
    
    self : np.array, bool, shape (N_best, n_features)
        N_best boolean masks for feature samples
    """

    N_features = X.shape[1]
    all_features = X.columns
    samples = np.random.binomial(1, p0, ( N_best, N_features)).astype(bool)
    scores = np.array([])
    for epoch in range(epochs):
        # crossing
        for i in range(N_best):
            j = np.random.randint(0, N_best)
            cross = np.random.binomial(1, p_cross, N_features).astype(bool)
            if (samples[i][cross] != samples[j][cross]).any():
                new_sample = np.copy(samples[i])
                new_sample[cross] = np.copy(samples[j][cross])
                if new_sample.any():
                    samples = np.vstack([samples, new_sample])

        # mutations
        for i in range(N_best, len(samples)):
            mutate = np.random.binomial(1, p_mut, N_features).astype(bool)
            samples[i][mutate] = ~samples[i][mutate]

        # new random samples
        for i in range(int(N_best*0.2)):
            new_sample = np.random.binomial(1, p0, N_features).astype(bool)
            samples = np.vstack([samples, new_sample])
        for i, sample in enumerate(samples[N_best:]):
            score = cross_val_score(estimator, X[all_features[sample]], 
                                    y, scoring=scoring, cv=cv).mean()
            scores = np.append(scores, score)
        ind_best = scores.argsort()[::-1]
        print 'epoch = ', epoch+1
        print scores[ind_best][:5]
        scores = scores[ind_best][:N_best]
        samples = samples[ind_best]
        samples = samples[:N_best]
    return samples
