import bpy
import pathlib

import numpy as np
import scipy.io as sio

TEMP_DATA = {}

class VidyaPCASynthesizer(bpy.types.Operator):
    bl_idname = 'vidya.pcasynthesizer'
    bl_label = 'PCA Synthesizer'
    bl_description = "Given the matrix file start synthesizing different shapes"
    bl_options = {'REGISTER', 'UNDO'}
    
    _mat_path: pathlib.Path
    _eigenvalues: np.ndarray
    _eigenvectors: np.ndarray
    _eigenratios: np.ndarray
    _mu: np.ndarray
    _transformed: np.ndarray
    _data: np.ndarray
    
    _mean_mesh: bpy.types.Object
    
    def _construct_mean_mesh(self, context, meandata: np.ndarray)->bpy.types.Object:
        mesh_name: str = self._mat_path.stem
        current_mean_mesh: bpy.types.Object = context.view_layer.objects.get(mesh_name, None)
        if(not current_mean_mesh):
            mesh_data = bpy.types.Mesh.new(mesh_name)
            
    
    def execute(self, context):
        mat_path: pathlib.Path = pathlib.Path(bpy.path.abspath(context.scene.VIDYA_PCA_Matrix))
        cache: dict = TEMP_DATA.get(mat_path.stem, None)
        self._mat_path = mat_path
        if(not cache):
            if(not mat_path.exists()):
                self.report({'WARNING'}, f'The given mat file path {mat_path} does not exist')
                return {'FINISHED'}
            try:
                mat_file: dict = sio.loadmat(f'{mat_path.resolve()}')
                TEMP_DATA[mat_path.stem] = mat_file
                cache = mat_file
            except ValueError:
                self.report({'WARNING'}, f'The given mat file path {mat_path} is not valid')
                return {'FINISHED'}
        
        self._mat = cache
        self._eigenratios = self._mat.get('eigenratios')
        self._eigenvalues = self._mat.get('eigenvalues')
        self._eigenvectors = self._mat.get('eigenvectors')
        self._mu = self._mat.get('mu')
        self._transformed = self._mat.get('transformed')
        self._data = self._mat.get('X')
        self._mean_mesh = self._construct_mean_mesh()
        return {'FINISHED'}