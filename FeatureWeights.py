import numpy as np
import scipy.io as sio

def synthesizeEigenVectors(eigenvectors, eigenvalues, coeffs, factor):

    K = eigenvalues.shape[1]
    sum_e_vectors = np.zeros((eigenvectors.shape[0], 1))

    eigenvectors = eigenvectors.T
    weights = np.diag(coeffs.flatten())
    eigenvalues = np.diag(np.abs(eigenvalues.flatten())**0.5)

    e_vectors = (weights@eigenvalues)@eigenvectors

    sum_e_vectors = np.sum(e_vectors, axis=0);    
    sum_e_vectors.shape = (int(sum_e_vectors.shape[0] / 3), 3);

    return sum_e_vectors, e_vectors

matrix: dict = sio.loadmat('./matrices/all_mats_sklearn.mat')

eigenvalues: np.ndarray = matrix['eigenvalues']
eigenvectors: np.ndarray = matrix['eigenvectors']
eigenratios: np.ndarray = matrix['eigenratios']
phenotypeParameters: np.ndarray = matrix['phenotypeParameters']
transformed: np.ndarray = matrix['transformed']
mu: np.ndarray = matrix['mu']
K: int = eigenvalues.shape[1]
L: int = phenotypeParameters.shape[1] + 1#3 features and it all adds up to 1 so totally 4 values. 3 for feature weights and the last value is 1;
features_matrix: np.ndarray = np.ones((K, L))
features_matrix[:,0:3] = phenotypeParameters[:K,0:3]


M_Matrix: np.ndarray = transformed.dot(np.linalg.pinv(features_matrix.T))

selection: int = 100
select_feature: np.ndarray = np.copy(features_matrix[selection])
select_feature[-1] = 0.0
select_feature.shape = (select_feature.shape[0], 1)

delta_P = M_Matrix.dot(select_feature)
delta_P = delta_P.T


# print('='*80);
print(delta_P);
# print('='*80);
print(transformed[selection])

print(transformed.shape)