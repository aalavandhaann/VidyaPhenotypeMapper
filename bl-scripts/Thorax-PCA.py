# context.area: VIEW_3D
import bpy
import bmesh
import sys
import pathlib

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA as sklearnPCA

class VertexGroup():
    vertices: list = []
    faces: list[list[int]] = []
    count: int = 0
    facescount: int = 0
    name: str = None

    def __init__(self) -> None:
        self.vertices = []
        self.faces = []
        self.count = 0

    def addVertexIndex(self, index: int)->None:
        self.vertices.append(index)
        self.count = len(self.vertices)
    
    def addFaceIndices(self, indices: list[list])->None:
        self.faces.append(indices)
        self.facescount = len(self.faces)

    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Name: {self.name}, Count: #{self.count}, Faces: #{self.faces}'

class PCAAnalyzer():
    _minAge: float = 2.0
    _maxAge: float = 80.0

    _K: int

    _thoraxVertexIds: list[int]
    _thoraxFaceIndices: list[list[int]]

    _genders: np.ndarray
    _ages: np.ndarray
    _obesity: np.ndarray
    _heights: np.ndarray
    _muscularities: np.ndarray

    _phenotypeParameters: np.ndarray

    _eigenvectors: np.ndarray
    _eigenvalues: np.ndarray
    _eigenratios: np.ndarray
    _transformed: np.ndarray
    _mu: np.ndarray 
    _X: np.ndarray
    _XMinusMu: np.ndarray


    _mesh: bpy.types.Object
    _context: bpy.types.Context
    _thorax_group: VertexGroup

    _topology_path: pathlib.Path
    
    def __init__(
            self, context: bpy.types.Context, mesh: bpy.types.Object, *, 
            K = 9, 
            gender_division: int = 3, age_division: int=7, 
            obesity_division: int=3, heights_division: int = 3, muscularity_division: int = 3,
            topology_path: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('matrices').joinpath('thorax_topology.mat')) -> None:
        self._context = context
        self._mesh = mesh
        self._topology_path = topology_path
        
        if(not self._topology_path.exists()):
            raise FileNotFoundError(f'Topology mat file {self._topology_path.resolve()} was not found')
        
        self._K = K

        self._genders = np.linspace(0.0, 1.0, gender_division)
        self._ages = np.linspace(0.0, 1.0, age_division)

        self._obesity = np.linspace(0.0, 1.0, obesity_division)
        self._heights = np.linspace(0.0, 1.0, heights_division)
        self._muscularities = np.linspace(0.0, 1.0, muscularity_division)

        self._thorax_group = self._create_vertex_group_table(self._context, human, ['Thorax']).get('Thorax')      

    
    def _get_evaluated_mesh(self, context: bpy.types.Context, mesh: bpy.types.Object)->bpy.types.Object:
        depsgraph = context.evaluated_depsgraph_get()
        depsgraph.update()        
        #Get the evaluated mesh based on the depsgraph for correct positions
        mesh = mesh.evaluated_get(depsgraph)
        return mesh
    
    def _get_vertexgroup_coordinates(self, context: bpy.types.Context, mesh: bpy.types.Object, vgroup: VertexGroup, 
                                     gender: float, age: float, 
                                     obesity: float, height: float, muscularity: float)->np.ndarray:
        scene: bpy.types.Scene = context.scene
        
        scene.mpfb_macropanel_gender = gender
        scene.mpfb_macropanel_age = age
        
        scene.mpfb_macropanel_weight = obesity
        scene.mpfb_macropanel_height = height

        scene.mpfb_macropanel_proportions = 0.5
        scene.mpfb_macropanel_muscle = muscularity

        mesh_evaluated: bpy.types.Object = self._get_evaluated_mesh(context, mesh)
        vertices: np.ndarray = np.zeros((len(vgroup.vertices), 3))
        for i, vid in enumerate(vgroup.vertices):
            vertex = mesh_evaluated.data.vertices[vid]
            vertices[i] = vertex.co.to_tuple()

        return vertices

    '''
        Ensure to save the topology information to a matrix file for all the required vertex groups i.e vertex ids and the faces/polygons they make.
        The topolgy mat file should have [VertexGroupName].vertexIds and [VertexGroupName].faceIndices
    '''
    def _create_vertex_group_table(self, context: bpy.types.Context, mesh: bpy.types.Object, for_vertex_group_names: list[str] = []) -> dict:                  
        table: dict = {}    
        for_vertex_groups: list = []
        topology_mat: dict = sio.loadmat(self._topology_path)       

        if(not len(for_vertex_group_names)):
            for_vertex_groups = [vg for vg in mesh.vertex_groups]
        else:
            for_vertex_groups = [mesh.vertex_groups.get(vname, None) for vname in for_vertex_group_names]

        for vg in for_vertex_groups:
            if(not vg):
                continue
            vgroup = VertexGroup()
            vgroup.name = vg.name
            group_vertices: list[int] = topology_mat.get(f'{vg.name}.vertexIds', np.zeros((0)))
            group_faces: list[list[int]] = topology_mat.get(f'{vg.name}.faceIndices', np.zeros((0, 0)))

            if(not group_vertices.shape[0]):
                raise ValueError(f'The given mat file {self._topology_path} does not contain vertices information for {vg.name}')
            if(not group_faces.shape[0]):
                raise ValueError(f'The given mat file {self._topology_path} does not contain faces connectivity information for {vg.name}')
            
            group_vertices = group_vertices[0]            

            for vid in group_vertices:
                vgroup.addVertexIndex(vid)
            
            for face in group_faces:
                vgroup.addFaceIndices(face.tolist())

            table[vg.name] = vgroup

        return table
    
    def _pca_analyzer(self, X: np.ndarray, K: int = 0):
        if(not K):
            K = X.shape[0]
        X_std: np.ndarray = X#StandardScaler().fit_transform(X)
        sklearn_pca: sklearnPCA = sklearnPCA(n_components=K)
        Y_sklearn: np.ndarray = sklearn_pca.fit_transform(X_std)
        
        mu: np.ndarray = sklearn_pca.mean_
        mu.shape = (mu.shape[0], 1)
        D: np.ndarray = sklearn_pca.explained_variance_
        D_ratio: np.ndarray = sklearn_pca.explained_variance_ratio_
        V: np.ndarray = sklearn_pca.components_
        print('*'*40)
        print(f'PCA SIZE: {K} ')
        print('DATA ENTRIES SHAPE ::: ', X.shape)
        print('MEAN MATRIX SHAPE ::: ', mu.shape)
        print('EIGEN VALUES SHAPE ::: ', D.shape)
        print('EIGEN VECTORS SHAPE ::: ', V.shape)
        print('TRANSFORMED SHAPE ::: ', Y_sklearn.shape)

        self._eigenvalues = D
        self._eigenvectors = V.T
        self._eigenratios = D_ratio
        self._mu = mu
        self._X = X_std.T
        self._XMinusMu = (X_std.T - mu)
        self._transformed = Y_sklearn
        # sio.savemat(bpy.path.abspath('//matrices/all_mats_sklearn.mat'), {'eigenvectors':V.T, 'eigenvalues':D, 'mu':mu, 'X':X_std.T,'XMinusMu':(X_std.T - mu), 'transformed':Y_sklearn}) 
        return mu
    
    def evaluate(self)->None:
        parameters_as_indices: list = [[i for i in range(a.shape[0])] for a in [self._genders, self._ages, 
                                                                                self._obesity, self._heights, self._muscularities]]
        combinations = np.array(np.meshgrid(*parameters_as_indices)).T.reshape(-1, len(parameters_as_indices))
        values: np.ndarray = np.zeros(combinations.shape)
        X: list = []
        
        values[:, 0] = self._genders[combinations[:, 0]]
        values[:, 1] = self._ages[combinations[:, 1]]
        values[:, 2] = self._obesity[combinations[:, 2]]
        values[:, 3] = self._heights[combinations[:, 3]]
        values[:, 4] = self._muscularities[combinations[:, 4]]

        thorax_group: VertexGroup = self._thorax_group#self._create_vertex_group_table(human, ['Thorax']).get('Thorax')
        for (gender, age, obesity, height, muscularity) in values:
            vertices: np.ndarray = self._get_vertexgroup_coordinates(self._context, human, thorax_group, 
                                                                     gender, age, 
                                                                     obesity, height, muscularity).flatten()
            X.append(vertices)
        
        if(self._K == -1):
            self._K = values.shape[0]

        self._thoraxVertexIds = thorax_group.vertices
        self._thoraxFaceIndices = thorax_group.faces
        self._phenotypeParameters = values

        X = np.array(X)
        # mu = self._pca_analyzer(X, K=values.shape[1])
        mu = self._pca_analyzer(X, K=self._K)
        mu.shape = (int(mu.shape[0]/3), 3)
    
    def save(self, path: pathlib.Path)->None:
        sio.savemat(path, {
            'eigenvectors':self._eigenvectors, 
            'eigenvalues':self._eigenvalues, 
            'eigenratios': self._eigenratios,
            'mu':self._mu, 
            'X':self._X,
            'XMinusMu':self._XMinusMu, 
            'transformed': self._transformed,
            'phenotypeParameters':self._phenotypeParameters,
            'vertexIds':self._thoraxVertexIds,
            'faceIndices': self._thoraxFaceIndices
            }) 
    
    @property
    def thorax_group(self) -> VertexGroup:
        return self._thorax_group

if __name__ == '__main__':
    C: bpy.types.Context = bpy.context
    human: bpy.types.Object = C.view_layer.objects.get('Human', None)
    thorax_topology: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('matrices').joinpath('thorax_topology.mat')

    if(not thorax_topology.exists()):
        print('Thorax topology is needed before saving the PCA matrices')
        sys.exit(0)   

    if(not human):
        print('No MakeHuman template available. Ensure to add the template under the name "Human" first')
        sys.exit(0)   


    # pcaAnalyzer: PCAAnalyzer = PCAAnalyzer(C, human, obesity_division=5)
    pcaAnalyzer: PCAAnalyzer = PCAAnalyzer(C, human, obesity_division=3, age_division=20, K = -1)    
    pcaAnalyzer.evaluate()
    pcaAnalyzer.save(bpy.path.abspath('//matrices/all_mats_sklearn.mat'))
       
        

    