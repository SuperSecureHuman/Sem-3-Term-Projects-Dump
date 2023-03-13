import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

def softmax(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)
    y = np.exp(x)
    return y / np.sum(y, axis=axis, keepdims=True)

def loss(X, Y, W):
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) +
                  np.sum(np.log(np.sum(np.exp(Z), axis=1)))) # Since the Loss works on the output of the model, It expects the Y to be onehot encoded
    return loss

def gradient_l2(X, Y, W, mu):

    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W    # Since the Loss works on the output of the model, It expects the Y to be onehot encoded
    return gd
    
def gradient_l1(X, Y, W, mu):
    """
    Y: onehot encoded 
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + mu * np.sign(W)
    return gd
    

gradient = gradient_l1

def plot_decision_boundry(X, Y, weight, save=False, name=""):
    """
    X: 2D
    Y: 1D
    W: 2D
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z =  -np.c_[xx.ravel(), yy.ravel()] @ weight
    Z = softmax(Z, axis=1)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    if save:
        plt.savefig(name)
        plt.close()
    else:
        return

def gradient_descent(X, Y, max_iter=1000, eta=0.05, mu=0.01, save_plot=True):
    """
    X: N x D
    Y: N x 1
    """
    # One-hot encode the labels
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
    
    # Initialize the weights to zeros
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    
    # Track the steps, weights, and losses
    step = 0
    step_lst = [] 
    loss_lst = []
    W_lst = []
    
    # Iterate until the max number of iterations
    while step < max_iter:
        step += 1
        
        # Update the weights
        W -= eta * gradient(X, Y_onehot, W, mu)
        
        # Track the step, weights, and loss
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))
        if save_plot:
            name = "plots/" + str(step) + ".jpg"
            plot_decision_boundry(X, Y, W, save=True, name=name)

    # Save the steps, weights, and losses
    df = pd.DataFrame({
        'step': step_lst, 
        'loss': loss_lst
    })
    return df, W

class Multiclass:
    def fit(self, X, Y):
        self.loss_steps, self.W = gradient_descent(X, Y, mu=0)

    def loss_plot(self):
        return self.loss_steps.plot(
            x='step', 
            y='loss',
            xlabel='step',
            ylabel='loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)
    
    def accuracy(self, H, Y):
        return np.mean(self.predict(H) == Y)

    def confusion_matrix(self, H, Y):
        return pd.crosstab(
            Y, 
            self.predict(H), 
            rownames=['Actual'], 
            colnames=['Predicted']
        )

    def loss(self, X, Y):
        Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
        return loss(X, Y_onehot, self.W)
    
X = load_iris().data
Y = load_iris().target

# Convert iris into 2 feature 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# fit model
model = Multiclass()
model.fit(X_pca, Y)
print(model.accuracy(X_pca, Y))