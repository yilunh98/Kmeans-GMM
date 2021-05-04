import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import pairwise_distances  # Don't use other functions in sklearn


def train_kmeans(train_data, initial_centroids):

    # prev_err = 0
    for iter in range(50):
        distances = pairwise_distances(train_data, initial_centroids, metric='euclidean')
        classification = np.argmin(distances, axis=1)

        result = {}
        for j in range(16):
            result[j] = []
        for i in range(train_data.shape[0]):
            result[classification[i]].append(train_data[i])
        for c in range(16):
            initial_centroids[c] = np.average(result[c],axis=0)
        
        compressed_data = [initial_centroids[classification[i]] for i in range(train_data.shape[0])]
        err = calculate_error(train_data, np.array(compressed_data))
        print('error in iter %d is %f'%(iter, err))
        # if abs(err-prev_err)<1e-3:
        #     break
        # else:
        #     prev_err = err

           
    states = {
          'centroids': initial_centroids
      }
    return states

def test_kmeans(states, test_data):
    result = {}
    
    centers = states['centroids']
    distances = pairwise_distances(test_data, centers, metric='euclidean')
    classification = np.argmin(distances, axis=1)
    compressed_data = np.array([centers[classification[i]] for i in range(test_data.shape[0])])

    plt.imshow(compressed_data.reshape(512, 512, -1).astype(np.uint8))
    imageio.imwrite(r'compressed_kmeans.jpg', compressed_data.reshape(512, 512, -1).astype(np.uint8))
    result['pixel-error'] = calculate_error(test_data, compressed_data)
    return result

### DO NOT CHANGE ###
def calculate_error(data, compressed_data):
    assert data.shape == compressed_data.shape
    error = np.sqrt(np.mean(np.power(data - compressed_data, 2)))
    return error
### DO NOT CHANGE ###

# Load data
img_small = np.array(imageio.imread('q1data/mandrill-small.tiff')) # 128 x 128 x 3
img_large = np.array(imageio.imread('q1data/mandrill-large.tiff')) # 512 x 512 x 3

ndim = img_small.shape[-1]
train_data = img_small.reshape(-1, ndim).astype(float)
test_data = img_large.reshape(-1, ndim).astype(float)
imageio.imwrite(r'original.jpg', test_data.reshape(512, 512, -1).astype(np.uint8))

# K-means
num_centroid = 16
initial_centroid_indices = [16041, 15086, 15419,  3018,  5894,  6755, 15296, 11460, 
                            10117, 11603, 11095,  6257, 16220, 10027, 11401, 13404]
initial_centroids = train_data[initial_centroid_indices, :]
states = train_kmeans(train_data, initial_centroids)
result_kmeans = test_kmeans(states, test_data)
print('Kmeans result=', result_kmeans)


from scipy.stats import multivariate_normal  # Don't use other functions in scipy

def train_gmm(train_data, init_pi, init_mu, init_sigma):

    K = len(init_pi)
    N = train_data.shape[0]
    posterior = np.zeros((N,K))
    for iter in range(50):
        for k in range(K):
            posterior[:,k] = init_pi[k] * multivariate_normal.pdf(train_data, init_mu[k], init_sigma[k])
        
        gamma = posterior / posterior.sum(axis=1).reshape(-1, 1)   # E 
        init_pi = gamma.sum(axis=0)/N
        Nk = init_pi * N
        init_mu = gamma.T@train_data/Nk.reshape(-1,1)        

        for k in range(K):                                         # M 
            init_sigma[k] = (train_data-init_mu[k]).T@((train_data-init_mu[k])*gamma[:,k].reshape(-1,1))/Nk[k]

        log = np.sum(gamma@np.diag(np.log(init_pi)))
        print('log-likelihood in iteration %d is %f'%(iter, log))
    
    states = {
        'pi': init_pi,
        'mu': init_mu,
        'sigma': init_sigma,
    }
    return states

def test_gmm(states, test_data):
    result = {}
    pi, mu, sigma = states['pi'], states['mu'], states['sigma']

    K = len(pi)
    N = test_data.shape[0]
    posterior = np.zeros((N,K))
    for k in range(K):
        posterior[:,k] = pi[k] * multivariate_normal.pdf(test_data, mu[k], sigma[k])
    clf = np.argmax(posterior, axis=1)

    compressed_data = np.array([mu[clf[i]] for i in range(N)])

    plt.imshow(compressed_data.reshape(512, 512, -1).astype(np.uint8))
    imageio.imwrite(r'compressed_GMM.jpg', compressed_data.reshape(512, 512, -1).astype(np.uint8))
    result['pixel-error'] = calculate_error(test_data, compressed_data)
    return result

# GMM
num_centroid = 5
init_pi = np.ones((num_centroid, 1)) / num_centroid
init_mu = initial_centroids[:num_centroid, :]
init_sigma = np.tile(np.identity(ndim), [num_centroid, 1, 1])*1000.

states = train_gmm(train_data, init_pi, init_mu, init_sigma)
result_gmm = test_gmm(states, test_data)
print('GMM result=', result_gmm)

