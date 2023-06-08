import bpy
import pathlib
import numpy as np
from VidyaPCASynthesizer.utilities import get_cache_matrix, is_matrix_loaded, load_matrix_to_cache, get_cache_matrix_name

def loadMATFile(self, context):
    bpy.ops.vidya.pcasynthesizer('EXEC_DEFAULT')

def slider_update(self, context):
    mesh: bpy.types.Object = context.view_layer.objects.get(self.mesh_name)
    if(mesh):
        mesh.VIDYA_PCA_Data.update(context)

class VIDYAEigenSlider(bpy.types.PropertyGroup):
    mesh_name: bpy.props.StringProperty(name="Mesh Name", description="Mesh to which this property belongs to")
    name: bpy.props.StringProperty(name="Slider Name", description="Filename (or mesh object name)")
    coefficient: bpy.props.FloatProperty(name='Coefficient', description="Coefficient of the eigen value", subtype='FACTOR', min=0, max=1.0, update=slider_update, default=0.0)

class VIDYAPCAEigenData(bpy.types.PropertyGroup):    
    mat_file_name: bpy.props.StringProperty(name='Matrix File Name', description="Name of the matrix file", default='//')
    mat_file_path: bpy.props.StringProperty(name='Matrix File Path', description="Location of the matrix file", default='//', subtype="FILE_PATH")
    sliders: bpy.props.CollectionProperty(type=VIDYAEigenSlider)
    slider_index: bpy.props.IntProperty(name='Current Slider Index', description="Slider index to maintain selection in UIList", default=0)

    def createSliders(self)->None:
        self.sliders.clear()
        mat_dict: dict = get_cache_matrix(pathlib.Path(self.mat_file_path))
        eigen_values: np.ndarray = mat_dict.get('eigenvalues')
        
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

        print(f'E: {eigen_values_vectors.shape}, χ: {weights.shape}, χ.E = {e_vectors.shape}')
        print(eigenratios.shape)
        print(np.sum(eigenratios.ravel()))

        sum_e_vectors = mu + sum_e_vectors 

        for i, (x, y, z) in enumerate(sum_e_vectors):
            mesh.data.vertices[i].co = (x, y, z)


def register():
    bpy.types.Scene.VIDYA_PCA_Matrix = bpy.props.StringProperty(name="Library Blend file path", default="./mat.mat", description="Path to the linked library Blend file", subtype="FILE_PATH", update=loadMATFile)
    bpy.types.Object.VIDYA_PCA_Data = bpy.props.PointerProperty(name="PCA Data", description="Pointer to the PCA Data", type=VIDYAPCAEigenData)

def unregister():
    bpy.props.RemoveProperty(bpy.types.Scene, attr='VIDYA_PCA_Matrix')