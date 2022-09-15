# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np
import math

class ePL_KRLS_DISCO:
    def __init__(self, alpha = 0.001, beta = 0.05, lambda1 = 0.0000001, sigma = 0.5, tau = 0.05, omega = 1, e_utility = 0.05):
        self.hyperparameters = pd.DataFrame({'alpha':[alpha],'beta':[beta], 'lambda1':[lambda1], 'sigma':[sigma], 'tau':[tau], 'omega':[omega], 'e_utility':[e_utility]})
        self.parameters = pd.DataFrame(columns = ['Center', 'Dictionary', 'nu', 'P', 'Q', 'Theta','ArousalIndex', 'Utility', 'SumLambda', 'TimeCreation', 'CompatibilityMeasure', 'OldCenter', 'tau', 'lambda'])
        # Parameters used to calculate the utility measure
        self.epsilon = []
        self.eTil = [0.]
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
         
    def fit(self, X, y):
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Initialize the first rule
        self.Initialize_First_Cluster(x, y[0])
        for k in range(1, X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute the compatibility measure and the arousal index for all rules
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['ArousalIndex'].astype('float64').idxmin()
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'ArousalIndex'] > self.hyperparameters.loc[0, 'tau'] and self.ExcludedRule == 0:
                self.Initialize_Cluster(x, y[k], k+1, MaxIndexCompatibility)
            else:
                self.Rule_Update(x, y[k], MaxIndexCompatibility)
                self.KRLS(x, y[k], MaxIndexCompatibility)
            self.Lambda(x)
            if self.parameters.shape[0] > 1:
                self.Utility_Measure(X[k,], k+1)
            self.rules.append(self.parameters.shape[0])
            # Finding the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            # Computing the output
            Output = 0
            for ni in range(self.parameters.loc[MaxIndexCompatibility, 'Dictionary'].shape[1]):
                Output = Output + self.parameters.loc[MaxIndexCompatibility, 'Theta'][ni] * self.Kernel_Gaussiano(self.parameters.loc[MaxIndexCompatibility, 'Dictionary'][:,ni].reshape(-1,1), x)
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            # Updating epsilon and e_til
            self.epsilon.append(math.exp(-0.5) * (2/(math.exp(-0.8 * self.eTil[-1] - abs(Output - y[k]))) - 1))
            self.eTil.append(0.8 * self.eTil[-1] + abs(Output - y[k]))
        return self.OutputTrainingPhase, self.rules
            
    def predict(self, X):
        for k in range(X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Computing the compatibility measure
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
            # Finding the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            # Computing the output
            Output = 0
            for ni in range(self.parameters.loc[MaxIndexCompatibility, 'Dictionary'].shape[1]):
                Output = Output + self.parameters.loc[MaxIndexCompatibility, 'Theta'][ni] * self.Kernel_Gaussiano(self.parameters.loc[MaxIndexCompatibility, 'Dictionary'][:,ni].reshape(-1,1), x)
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase
        
    def Initialize_First_Cluster(self, x, y):
        Q = np.linalg.inv(np.ones((1,1))*(self.hyperparameters.loc[0, 'lambda1'] + (self.Kernel_Gaussiano(x, x))))
        Theta = Q*y
        self.parameters = pd.DataFrame([[x, x, self.hyperparameters['sigma'][0], np.ones((1,1)), Q, Theta, 0., 1., 0., 1., 1., 1., np.zeros((x.shape[0],1)), 1.]], columns = ['Center', 'Dictionary', 'nu', 'P', 'Q', 'Theta', 'ArousalIndex', 'Utility', 'SumLambda', 'NumObservations', 'TimeCreation', 'CompatibilityMeasure', 'OldCenter', 'tau'])
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
    
    def Initialize_Cluster(self, x, y, k, i):
        Q = np.linalg.inv(np.ones((1,1))*(self.hyperparameters.loc[0, 'lambda1'] + (self.Kernel_Gaussiano(x, x))))
        Theta = Q*y
        nu = (np.linalg.norm(x - self.parameters.loc[i, 'Center'])/math.sqrt(-2 * np.log(max(self.epsilon))))
        NewRow = pd.DataFrame([[x, x, nu, np.ones((1,1)), Q, Theta, 0., 1., 0., 1., k, 1., np.zeros((x.shape[0],1)), 1.]], columns = ['Center', 'Dictionary', 'nu', 'P', 'Q', 'Theta', 'ArousalIndex', 'Utility', 'SumLambda', 'NumObservations', 'TimeCreation', 'CompatibilityMeasure', 'OldCenter', 'tau'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
    
    def Kernel_Gaussiano(self, Vector1, Vector2):
        return math.exp(-((((np.linalg.norm(Vector1-Vector2))**2)/(2*self.hyperparameters.loc[0, 'sigma']**2))))
    
    def Compatibility_Measure(self, x, i):
        if np.isnan(np.corrcoef(self.parameters.loc[i, 'Center'].T, x.T)[0,1]):
            self.parameters.at[i, 'CompatibilityMeasure'] = (1 - ((np.linalg.norm(x - self.parameters.loc[i, 'Center']))/x.shape[0]))
        else:
            self.parameters.at[i, 'CompatibilityMeasure'] = (1 - ((np.linalg.norm(x - self.parameters.loc[i, 'Center']))/x.shape[0])) * ( ( np.corrcoef(self.parameters.loc[i, 'Center'].T, x.T)[0,1] + 1) / 2 )
            
    def Arousal_Index(self, i):
        self.parameters.at[i, 'ArousalIndex'] = self.parameters.loc[i, 'ArousalIndex'] + self.hyperparameters.loc[0, 'beta'] * (1 - self.parameters.loc[i, 'CompatibilityMeasure'] - self.parameters.loc[i, 'ArousalIndex'])
    
    def Rule_Update(self, x, y, i):
        # Update the number of observations in the rule
        self.parameters.at[i, 'NumObservations'] = self.parameters.loc[i, 'NumObservations'] + 1
        # Store the old cluster center
        self.parameters.at[i, 'OldCenter'] = self.parameters.loc[i, 'Center']
        # Update the cluster center
        self.parameters.at[i, 'Center'] = self.parameters.loc[i, 'Center'] + (self.hyperparameters.loc[0, 'alpha'] * (self.parameters.loc[i, 'CompatibilityMeasure'])**(1 - self.parameters.loc[i, 'ArousalIndex'])) * (x - self.parameters.loc[i, 'Center'])
                       
    def Lambda(self, x):
        for row in self.parameters.index:
            self.parameters.at[row, 'lambda'] = self.parameters.loc[row, 'tau'] / sum(self.parameters['tau'])
            self.parameters.at[row, 'SumLambda'] = self.parameters.loc[row, 'SumLambda'] + self.parameters.loc[row, 'lambda']
            
    def Utility_Measure(self, x, k):
        # Calculating the utility
        remove = []
        for i in self.parameters.index:
            if (k - self.parameters.loc[i, 'TimeCreation']) == 0:
                self.parameters.at[i, 'Utility'] = 1
            else:
                self.parameters.at[i, 'Utility'] = self.parameters.loc[i, 'SumLambda'] / (k - self.parameters.loc[i, 'TimeCreation'])
            if self.parameters.loc[i, 'Utility'] < self.hyperparameters.loc[0, 'e_utility']:
                remove.append(i)
        if len(remove) > 0:    
            self.parameters = self.parameters.drop(remove)
            # Stoping to creating new rules when the model exclude the first rule
            self.ExcludedRule = 1
            
    def KRLS(self, x, y, i):
        # Update the kernel size
        self.parameters.at[i, 'nu'] = math.sqrt((self.parameters.loc[i, 'nu'])**2 + (((np.linalg.norm(x - self.parameters.loc[i, 'Center']))**2 - (self.parameters.loc[i, 'nu'])**2)/self.parameters.loc[i, 'NumObservations']) + ((self.parameters.loc[i, 'NumObservations'] - 1) * ((np.linalg.norm(self.parameters.loc[i, 'Center'] - self.parameters.loc[i, 'OldCenter']))**2))/self.parameters.loc[i, 'NumObservations'])
        # Compute g
        g = np.array(())
        for ni in range(self.parameters.loc[i, 'Dictionary'].shape[1]):
            g = np.append(g, [self.Kernel_Gaussiano(self.parameters.loc[i, 'Dictionary'][:,ni].reshape(-1,1), x)])
        G = g.reshape(g.shape[0],1)
        # Computing z
        z = np.matmul(self.parameters.loc[i, 'Q'], g)
        Z = z.reshape(z.shape[0],1)
        # Computing r
        r = self.hyperparameters.loc[0, 'lambda1'] + 1 - np.matmul(Z.T, g).item()
        # if r == 0:
        #     r = self.hyperparameters.loc[0, 'lambda1']
        # Estimating the error
        EstimatedError = y - np.matmul(G.T, self.parameters.loc[i, 'Theta'])
        # Searching for the lowest distance between the input and the dictionary inputs
        distance = []
        for ni in range(self.parameters.loc[i, 'Dictionary'].shape[1]):
            distance.append(np.linalg.norm(self.parameters.loc[i, 'Dictionary'][:,ni].reshape(-1,1) - x))
        # Finding the index of minimum distance
        IndexMinDistance = np.argmin(distance)
        # Novelty criterion
        if distance[IndexMinDistance] > 0.1 * self.parameters.loc[i, 'nu']:
            self.parameters.at[i, 'Dictionary'] = np.hstack([self.parameters.loc[i, 'Dictionary'], x])
            # Updating Q                      
            self.parameters.at[i, 'Q'] = (1/r)*(self.parameters.loc[i, 'Q']*r + np.matmul(Z,Z.T))
            self.parameters.at[i, 'Q'] = np.lib.pad(self.parameters.loc[i, 'Q'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeQ = self.parameters.loc[i, 'Q'].shape[0] - 1
            self.parameters.at[i, 'Q'][sizeQ,sizeQ] = (1/r)*self.hyperparameters.loc[0, 'omega']
            self.parameters.at[i, 'Q'][0:sizeQ,sizeQ] = (1/r)*(-z)
            self.parameters.at[i, 'Q'][sizeQ,0:sizeQ] = (1/r)*(-z)
            # Updating P
            self.parameters.at[i, 'P'] = np.lib.pad(self.parameters.loc[i, 'P'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeP = self.parameters.loc[i, 'P'].shape[0] - 1
            self.parameters.at[i, 'P'][sizeP,sizeP] = self.hyperparameters.loc[0, 'omega']
            # Updating Theta
            self.parameters.at[i, 'Theta'] = self.parameters.loc[i, 'Theta'] - (Z*(1/r)*EstimatedError)
            self.parameters.at[i, 'Theta'] = np.vstack([self.parameters.loc[i, 'Theta'],(1/r)*EstimatedError])
        else:
            # Calculating q
            q = np.matmul(self.parameters.loc[i, 'P'], Z)/(1 + np.matmul(np.matmul(Z.T, self.parameters.loc[i, 'P']), Z))
            # Updating P
            self.parameters.at[i, 'P'] = self.parameters.loc[i, 'P'] - (np.matmul(np.matmul(np.matmul(self.parameters.loc[i, 'P'],Z), Z.T), self.parameters.loc[i, 'P']))/(1 + np.matmul(np.matmul(Z.T, self.parameters.loc[i, 'P']), Z))
            # Updating Theta
            self.parameters.at[i, 'Theta'] = self.parameters.loc[i, 'Theta'] + np.matmul(self.parameters.loc[i, 'Q'], q) * EstimatedError