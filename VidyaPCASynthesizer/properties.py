import bpy
import time
import pathlib
import numpy as np
from VidyaPCASynthesizer.utilities import get_cache_matrix, is_matrix_loaded, load_matrix_to_cache, get_cache_matrix_name

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


# from numpy.linalg import pinv
from scipy.linalg import pinv

def loadMATFile(self, context):
    bpy.ops.vidya.pcasynthesizer('EXEC_DEFAULT')

def eigen_slider_update(self, context):
    mesh: bpy.types.Object = context.view_layer.objects.get(self.mesh_name)
    if(mesh and not self.soft_update):
        mesh.VIDYA_PCA_Data.update(context)

def feature_slider_update(self, context):
    mesh: bpy.types.Object = context.view_layer.objects.get(self.mesh_name)
    if(mesh and not self.soft_update):
        mesh.VIDYA_PCA_Features.update(context)

class VIDYAEigenSlider(bpy.types.PropertyGroup):
    mesh_name: bpy.props.StringProperty(name="Mesh Name", description="Mesh to which this property belongs to")
    name: bpy.props.StringProperty(name="Slider Name", description="Filename (or mesh object name)")
    coefficient: bpy.props.FloatProperty(name='Coefficient', description="Coefficient of the eigen value", subtype='FACTOR', min=-10.0, max=10.0, update=eigen_slider_update, default=0.0)
    soft_update: bpy.props.BoolProperty(name='Soft Update', description="To control speed of sliders", default=False)

class VIDYAFeatureSlider(bpy.types.PropertyGroup):
    mesh_name: bpy.props.StringProperty(name="Mesh Name", description="Mesh to which this property belongs to")
    name: bpy.props.StringProperty(name="Slider Name", description="Filename (or mesh object name)")
    coefficient: bpy.props.FloatProperty(name='Coefficient', description="Coefficient of the feature", subtype='FACTOR', min=0.0, max=1.0, update=feature_slider_update, default=0.0)
    soft_update: bpy.props.BoolProperty(name='Soft Update', description="To control speed of sliders", default=False)

class VIDYAPCAEigenData(bpy.types.PropertyGroup):    
    mat_file_name: bpy.props.StringProperty(name='Matrix File Name', description="Name of the matrix file", default='//')
    mat_file_path: bpy.props.StringProperty(name='Matrix File Path', description="Location of the matrix file", default='//', subtype="FILE_PATH")
    sliders: bpy.props.CollectionProperty(type=VIDYAEigenSlider)
    slider_index: bpy.props.IntProperty(name='Current Slider Index', description="Slider index to maintain selection in UIList", default=0)

    def _get_evaluated_mesh(self, context: bpy.types.Context, mesh: bpy.types.Object)->bpy.types.Object:
        depsgraph = context.evaluated_depsgraph_get()
        depsgraph.update()        
        #Get the evaluated mesh based on the depsgraph for correct positions
        mesh = mesh.evaluated_get(depsgraph)
        return mesh

    def _get_makehuman_mesh(self, context)->bpy.types.Object:
        for o in context.view_layer.objects:
            if(o.type == 'MESH'):
                if(o.MPFB_HUM_is_human_project):
                    return o

        return None

    def createSliders(self)->None:
        
        mat_dict: dict = get_cache_matrix(pathlib.Path(self.mat_file_path))
        eigen_values: np.ndarray = mat_dict.get('eigenvalues')

        self.sliders.clear()
        for i in range(eigen_values.shape[1]):
            slider = self.sliders.add()
            slider.mesh_name = self.mat_file_name
    
    def update(self, context)->None:
        mat: dict = {}
        if(not is_matrix_loaded(self.mat_file_name)):
            mat = load_matrix_to_cache(pathlib.Path(self.mat_file_path))
        else:
            mat = get_cache_matrix_name(self.mat_file_name)
        
        coefficients = []
        for slider in self.sliders:
            coefficients.append(slider.coefficient)
        
        mesh: bpy.types.Object = context.view_layer.objects.get(self.mat_file_name)
        eigenvectors: np.ndarray = mat.get('eigenvectors')
        eigenvalues: np.ndarray = mat.get('eigenvalues')
        eigenratios: np.ndarray = mat.get('eigenratios')
        mu: np.ndarray = mat.get('mu')

        K: int = eigenvalues.shape[1]
        sum_e_vectors: np.ndarray = np.zeros((eigenvectors.shape[0], 1))

        eigenvectors: np.ndarray = eigenvectors.T        
        weights: np.ndarray = np.diag(coefficients)
        eigenvalues: np.ndarray = np.diag(np.abs(eigenvalues.flatten())**0.5)

        eigen_values_vectors: np.ndarray = eigenvalues@eigenvectors
        e_vectors: np.ndarray = weights@eigen_values_vectors
        sum_e_vectors = np.sum(e_vectors, axis=0);    
        sum_e_vectors.shape = (int(sum_e_vectors.shape[0] / 3), 3)      

        sum_e_vectors = mu + sum_e_vectors 

        for i, (x, y, z) in enumerate(sum_e_vectors):
            mesh.data.vertices[i].co = (x, y, z)
    
    def predict(self, context):
        mat: dict = {}
        vertices_pointer: list = []
        vertices: list[list[float]] = []
        N: int = 0
        if(not is_matrix_loaded(self.mat_file_name)):
            mat = load_matrix_to_cache(pathlib.Path(self.mat_file_path))
        else:
            mat = get_cache_matrix_name(self.mat_file_name)

        mu: np.ndarray = mat.get('mu').flatten()
        eigenvectors_mat: np.ndarray = mat.get('eigenvectors')
        eigenvalues_mat: np.ndarray = mat.get('eigenvalues')
        
        M:np.ndarray = mat.get('M', np.zeros((0, 0)))

        mesh: bpy.types.Object = self._get_makehuman_mesh(context)
        if(mesh):
            mesh = self._get_evaluated_mesh(context, mesh)
            vertexIds: list[int] = mat.get('vertexIds').astype(int).flatten().tolist()
            for vid in vertexIds:
                vertices_pointer.append(mesh.data.vertices[vid])

        else:
            mesh = context.view_layer.objects.get(self.mat_file_name)
            for v in mesh.data.vertices:
                vertices_pointer.append(v)

        N = len(vertices_pointer)
        for v in vertices_pointer:
            vertices.append([v.co.x, v.co.y, v.co.z])
        
        vertices = np.array(vertices).flatten()

        start = time.time()
        S_sum: np.array = np.diag(vertices - mu)
        lamuda_vectors: np.ndarray = eigenvectors_mat.T 
        lamuda: np.ndarray = np.diag(np.abs(eigenvalues_mat.flatten())**0.5)
        lamuda_product: np.ndarray = lamuda@lamuda_vectors

        S_sum_inv: np.ndarray = pinv(S_sum)

        W_inv: np.ndarray = lamuda_product@S_sum_inv# 
        W_full: np.ndarray = pinv(W_inv) 
        W: np.ndarray = np.sum(W_full, axis=0)    
        end = time.time()
        total_time = end - start
        print(f'TOTAL TIME: {total_time}')
        
        for i, slider in enumerate(self.sliders):
            slider.soft_update = True
            slider.coefficient = W[i]
            slider.soft_update = False
            
        self.update(context)
        if(M.shape[0]):
            print(M.shape)
            print(W.shape)
            # F = W@pinv(M)
            
        
        print(f'MATRIX SHAPES: \nS: {S_sum.shape}\nsqroot(λ).Λ: {lamuda_product.shape}\ninv(S_sum): {S_sum_inv.shape}\nW^(-1): {W_inv.shape}\nW Full: {W_full.shape}\nW: {W.shape}')

class VIDYAPCAFeatureData(bpy.types.PropertyGroup):    
    mat_file_name: bpy.props.StringProperty(name='Matrix File Name', description="Name of the matrix file", default='//')
    mat_file_path: bpy.props.StringProperty(name='Matrix File Path', description="Location of the matrix file", default='//', subtype="FILE_PATH")
    sliders: bpy.props.CollectionProperty(type=VIDYAFeatureSlider)
    slider_index: bpy.props.IntProperty(name='Current Slider Index', description="Slider index to maintain selection in UIList", default=0)
    
    def _get_evaluated_mesh(self, context: bpy.types.Context, mesh: bpy.types.Object)->bpy.types.Object:
        depsgraph = context.evaluated_depsgraph_get()
        depsgraph.update()        
        #Get the evaluated mesh based on the depsgraph for correct positions
        mesh = mesh.evaluated_get(depsgraph)
        return mesh

    def _get_makehuman_mesh(self, context)->bpy.types.Object:
        for o in context.view_layer.objects:
            if(o.type == 'MESH'):
                if(o.MPFB_HUM_is_human_project):
                    return o
                
    def createSliders(self)->None:
        
        mat_dict: dict = get_cache_matrix(pathlib.Path(self.mat_file_path))
        features: dict = mat_dict.get('labels', np.zeros((0, 0)))

        self.sliders.clear()
        for label in features:
            slider = self.sliders.add()
            slider.mesh_name = self.mat_file_name
            slider.name = label.upper()
    
    def update(self, context)->None:
        scene: bpy.types.Scene = context.scene
        mat: dict = {}
        if(not is_matrix_loaded(self.mat_file_name)):
            mat = load_matrix_to_cache(pathlib.Path(self.mat_file_path))
        else:
            mat = get_cache_matrix_name(self.mat_file_name)        

        import scipy.io as sio
        matt: dict = sio.loadmat(f'{pathlib.Path(self.mat_file_path)}')

        M: np.ndarray = mat.get('M')
        F: np.ndarray = np.ones((M.shape[1], 1))
        print(M.shape, F.shape)
        P_original: np.ndarray = mat.get('P')
        sliders: list[VIDYAFeatureSlider] = self.sliders
        
        for i, slider in enumerate(sliders):            
            F[i, 0] = slider.coefficient    

        P:np.ndarray = (M@F).flatten()
        print('P MATRIX : ', P.shape)
        mk_mesh: bpy.types.Object = self._get_makehuman_mesh(context)
        pca_mesh: bpy.types.Object = context.view_layer.objects.get(self.mat_file_name)

        pca_sliders: list[VIDYAEigenSlider] = pca_mesh.VIDYA_PCA_Data.sliders
        
        for i, pca_slider in enumerate(pca_sliders):
            pca_slider.soft_update = True
            pca_slider.coefficient =  P[i]
            pca_slider.soft_update = False

        pca_mesh.VIDYA_PCA_Data.update(context)

        bpy.ops.object.select_all(action="DESELECT")

        mk_mesh.select_set(True)
        context.view_layer.objects.active = mk_mesh
        scene.mpfb_macropanel_gender = F[0, 0]        
        scene.mpfb_macropanel_age = F[1, 0]
        scene.mpfb_macropanel_weight = F[2, 0]
        scene.mpfb_macropanel_height = F[3, 0]
        scene.mpfb_macropanel_muscle = F[4, 0]
        # scene.mpfb_macropanel_proportions = 0.5
        
        bpy.ops.object.select_all(action="DESELECT")
        pca_mesh.select_set(True)
        context.view_layer.objects.active = pca_mesh
        

def register():
    bpy.types.Scene.VIDYA_PCA_Matrix = bpy.props.StringProperty(name="Library Blend file path", default="./mat.mat", description="Path to the linked library Blend file", subtype="FILE_PATH", update=loadMATFile)
    bpy.types.Object.VIDYA_PCA_Data = bpy.props.PointerProperty(name="PCA Data", description="Pointer to the PCA Data", type=VIDYAPCAEigenData)
    bpy.types.Object.VIDYA_PCA_Features = bpy.props.PointerProperty(name="PCA Features", description="Pointer to the PCA Features", type=VIDYAPCAFeatureData)
    
def unregister():
    bpy.props.RemoveProperty(bpy.types.Scene, attr='VIDYA_PCA_Matrix')
    # bpy.props.RemoveProperty(bpy.types.Scene, attr='VIDYA_PCA_Data')
    # bpy.props.RemoveProperty(bpy.types.Scene, attr='VIDYA_PCA_Features')