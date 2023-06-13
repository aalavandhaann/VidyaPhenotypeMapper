import pathlib
import tqdm
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA as sklearnPCA

'''
The objective of this script is to find MF = P, given the PCA decomposition of multiple shapes (makehuman thoraxes).
Where, 
- P is a matrix of size M x k weights (M is the number of shapes or samples, k is the size of eigendecomposition)
- F is a matrix of size M x h containing phenotype or human understandable parameters (again M refers to number 
  of shapes or samples, and h refers to number of human understandable features)
- M is a matrix that is the tranformation from F to P
'''



def getPCAWeights(mu: np.ndarray, vertices: np.ndarray, eigenvalues_mat: np.ndarray, eigenvectors_mat: np.ndarray)->np.ndarray:
    S_sum: np.array = np.diag(np.array(vertices) - mu)

    lamuda_vectors: np.ndarray = eigenvectors_mat.T 
    lamuda: np.ndarray = np.diag(np.abs(eigenvalues_mat.flatten())**0.5)
    lamuda_product: np.ndarray = lamuda@lamuda_vectors
    S_sum_inv: np.ndarray = np.linalg.pinv(S_sum)
    W_inv: np.ndarray = lamuda_product@S_sum_inv
    W_full: np.ndarray = np.linalg.pinv(W_inv) 
    W: np.ndarray = np.sum(W_full, axis=0) 

    return W


if __name__ == '__main__':
    mat_file: pathlib.Path = pathlib.Path('matrices/all_mats_sklearn.mat')
    mat: dict = sio.loadmat(f'{mat_file}')

    original_data: np.ndarray = mat.get('X').T
    transformed: np.ndarray = mat.get('transformed')
    phenotypes_data: np.ndarray = mat.get('phenotypeParameters')
    mu: np.ndarray = mat.get('mu').flatten()
    eigenvalues_mat: np.ndarray = mat.get('eigenvalues')
    eigenvectors_mat: np.ndarray = mat.get('eigenvectors')
    
    F: np.ndarray = phenotypes_data.T
    P: np.ndarray = np.zeros((original_data.shape[0], eigenvalues_mat.shape[1]))

    for i, (gender, age, obesity) in tqdm.tqdm(enumerate(phenotypes_data), dynamic_ncols=True, desc='Computing PCA Weights:', colour='green'):
        vertices: np.ndarray = original_data[i]
        W: np.ndarray = getPCAWeights(mu, vertices, eigenvalues_mat, eigenvectors_mat)
        P[i] = W

    M = P@np.linalg.pinv(F)
    mat['M'] = M
    mat['P'] = P
    mat['F'] = F
    mat['labels'] = ['gender', 'age', 'obesity']
    sio.savemat(f'{mat_file}', mat)