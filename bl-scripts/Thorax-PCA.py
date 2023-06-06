# context.area: VIEW_3D
import bpy
import sys
import pathlib

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA as sklearnPCA

class VertexGroup():
    vertices: list = []
    count: int = 0
    name: str = None

    def __init__(self) -> None:
        self.vertices = []
        self.count = 0

    def addVertexIndex(self, index: int)->None:
        self.vertices.append(index)
        self.count = len(self.vertices)

    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Name: {self.name}, Count: #{self.count}'

class PCAAnalyzer():
    _minAge: float = 2.0
    _maxAge: float = 80.0

    _thoraxVertexIds: list[int]

    _genders: np.ndarray
    _ages: np.ndarray
    _obesity: np.ndarray

    _phenotypeParameters: np.ndarray

    _eigenvectors: np.ndarray
    _eigenvalues: np.ndarray
    _transformed: np.ndarray
    _mu: np.ndarray 
    _X: np.ndarray
    _XMinusMu: np.ndarray


    _mesh: bpy.types.Object
    

    def __init__(self, mesh: bpy.types.Object, *, age_division: int=7, obesity_division: int=5) -> None:
        self._mesh = mesh
        self._genders = np.linspace(0.0, 1.0, 3)
        self._ages = np.linspace(0.0, 1.0, age_division)
        self._obesity = np.linspace(0.0, 1.0, obesity_division)

    
    def _get_evaluated_mesh(self, context: bpy.types.Context, mesh: bpy.types.Object)->bpy.types.Object:
        depsgraph = context.evaluated_depsgraph_get()
        depsgraph.update()        
        #Get the evaluated mesh based on the depsgraph for correct positions
        mesh = mesh.evaluated_get(depsgraph)
        return mesh
    
    def _get_vertexgroup_coordinates(self, context: bpy.types.Context, mesh: bpy.types.Object, vgroup: VertexGroup, gender: float, age: float, obesity: float)->np.ndarray:
        scene: bpy.types.Scene = context.scene
        scene.mpfb_macropanel_weight = obesity
        scene.mpfb_macropanel_age = age
        scene.mpfb_macropanel_gender = gender
        scene.mpfb_macropanel_muscle = 0.0

        mesh_evaluated: bpy.types.Object = self._get_evaluated_mesh(context, mesh)
        vertices: np.ndarray = np.zeros((len(vgroup.vertices), 3))
        for i, vid in enumerate(vgroup.vertices):
            vertex = mesh_evaluated.data.vertices[vid]
            vertices[i] = vertex.co.to_tuple()

        return vertices

    def _create_vertex_group_table(self, mesh: bpy.types.Object, for_vertex_group_names: list[str] = []) -> dict:
        table: dict = {}    
        for_vertex_groups: list = []
        if(not len(for_vertex_group_names)):
            for_vertex_groups = [vg for vg in mesh.vertex_groups]
        else:
            for_vertex_groups = [mesh.vertex_groups.get(vname, None) for vname in for_vertex_group_names]

        for vg in for_vertex_groups:
            if(not vg):
                continue
            vgroup = VertexGroup()
            vgroup.name = vg.name
            for v in mesh.data.vertices:
                gids = [g.group for g in v.groups]
                if(vg.index in gids):
                    weight = vg.weight(v.index)
                    if(weight > (1.0 - 1e-2)):
                        vgroup.addVertexIndex(v.index)
            table[vg.name] = vgroup
        return table
    
    def _pca_analyzer(self, X: np.ndarray):
        K: int = X.shape[0]
        X_std: np.ndarray = X#StandardScaler().fit_transform(X)
        sklearn_pca: sklearnPCA = sklearnPCA(n_components=K)
        Y_sklearn: np.ndarray = sklearn_pca.fit_transform(X_std)
        
        mu: np.ndarray = sklearn_pca.mean_
        mu.shape = (mu.shape[0], 1)
        D: np.ndarray = sklearn_pca.explained_variance_
        D_ratio: np.ndarray = sklearn_pca.explained_variance_ratio_
        V: np.ndarray = sklearn_pca.components_
        print('*'*40)
        print('DATA ENTRIES SHAPE ::: ', X.shape)
        print('MEAN MATRIX SHAPE ::: ', mu.shape)
        print('EIGEN VALUES SHAPE ::: ', D.shape)
        print('EIGEN VECTORS SHAPE ::: ', V.shape)
        print('TRANSFORMED SHAPE ::: ', Y_sklearn.shape)

        self._eigenvalues = D
        self._eigenvectors = V.T
        self._mu = mu
        self._X = X_std.T
        self._XMinusMu = (X_std.T - mu)
        self._transformed = Y_sklearn
        # sio.savemat(bpy.path.abspath('//matrices/all_mats_sklearn.mat'), {'eigenvectors':V.T, 'eigenvalues':D, 'mu':mu, 'X':X_std.T,'XMinusMu':(X_std.T - mu), 'transformed':Y_sklearn}) 
        return mu
    
    def evaluate(self)->None:
        parameters_as_indices: list = [[i for i in range(a.shape[0])] for a in [self._genders, self._ages, self._obesity]]
        combinations = np.array(np.meshgrid(*parameters_as_indices)).T.reshape(-1, len(parameters_as_indices))
        values: np.ndarray = np.zeros(combinations.shape)
        X: list = []
        
        values[:, 0] = self._genders[combinations[:, 0]]
        values[:, 1] = self._ages[combinations[:, 1]]
        values[:, 2] = self._obesity[combinations[:, 2]]

        thorax_group: VertexGroup = self._create_vertex_group_table(human, ['Thorax']).get('Thorax')
        for gender, age, obesity in values:
            vertices: np.ndarray = self._get_vertexgroup_coordinates(C, human, thorax_group, gender, age, obesity).flatten()
            X.append(vertices)
        
        self._thoraxVertexIds = thorax_group.vertices
        self._phenotypeParameters = values

        X = np.array(X)
        mu = self._pca_analyzer(X)
        mu.shape = (int(mu.shape[0]/3), 3)
    
    def save(self, path: pathlib.Path)->None:
        sio.savemat(path, {
            'eigenvectors':self._eigenvectors, 
            'eigenvalues':self._eigenvalues, 
            'mu':self._mu, 
            'X':self._X,
            'XMinusMu':self._XMinusMu, 
            'transformed': self._transformed,
            'phenotypeParameters':self._phenotypeParameters,
            'vertexIds':self._thoraxVertexIds
            }) 
    

if __name__ == '__main__':
    C: bpy.types.Context = bpy.context
    human: bpy.types.Object = C.view_layer.objects.get('Human', None)

    if(not human):
        print('No MakeHuman template available. Ensure to add the template under the name "Human" first')
        sys.exit(0)

    pcaAnalyzer: PCAAnalyzer = PCAAnalyzer(human, obesity_division=5)
    pcaAnalyzer.evaluate()
    pcaAnalyzer.save(bpy.path.abspath('//matrices/all_mats_sklearn.mat'))