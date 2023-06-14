import bpy, bmesh
from mathutils import Vector
import pathlib

import numpy as np
import scipy.io as sio

from VidyaPCASynthesizer.utilities import get_cache_matrix

class VidyaPCAPredictor(bpy.types.Operator):
    bl_idname = 'vidya.pcapredictor'
    bl_label = 'PCA Predictor'
    bl_description = "Given the Sample data predict the features"
    bl_options = {'REGISTER', 'UNDO'}


    def execute(self, context):
        mat_path: pathlib.Path = pathlib.Path(bpy.path.abspath(context.scene.VIDYA_PCA_Matrix))
        mesh: bpy.types.Object = context.view_layer.objects.get(mat_path.stem)
        mesh.VIDYA_PCA_Data.predict(context)
        return {'FINISHED'}

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
    
    def _construct_mean_mesh(self, context, meanvertexpositions: np.ndarray, vertexids: list[int], faces: list[list[int]])->bpy.types.Object:
        collection: bpy.types.Collection = bpy.data.collections.get('PCA-Collection', bpy.data.collections.new('PCA-Collection'))
        mesh_name: str = self._mat_path.stem
        current_mesh_data: bpy.types.Mesh = bpy.data.meshes.get(mesh_name, bpy.data.meshes.new(mesh_name))
        current_mesh: bpy.types.Object = context.view_layer.objects.get(mesh_name, bpy.data.objects.new(mesh_name, current_mesh_data))

        bm = bmesh.new()        
        bm.from_mesh(current_mesh_data)
        bm.clear()

        for i, _ in enumerate(vertexids):
            x, y, z = meanvertexpositions[i]
            v = bm.verts.new(Vector((x, y, z)))
        
        bm.verts.ensure_lookup_table()
        
        for vindices in faces:
            faceverts = [bm.verts[vertexids.index(vindex)] for vindex in vindices]
            face = bm.faces.new(faceverts)

        bm.to_mesh(current_mesh_data)
        bm.free()

        if(not collection.objects.get(mesh_name)):
            collection.objects.link(current_mesh)

        if(not context.scene.collection.children.get(collection.name)):
            context.scene.collection.children.link(collection) 
        return current_mesh
            
    
    def execute(self, context):
        mat_path: pathlib.Path = pathlib.Path(bpy.path.abspath(context.scene.VIDYA_PCA_Matrix))

        self._mat_path = mat_path
        self._mat = get_cache_matrix(self._mat_path)
        self._eigenratios = self._mat.get('eigenratios')
        self._eigenvalues = self._mat.get('eigenvalues')
        self._eigenvectors = self._mat.get('eigenvectors')
        self._mu = self._mat.get('mu')
        self._transformed = self._mat.get('transformed')
        self._data = self._mat.get('X')

        self._mean_mesh = self._construct_mean_mesh(context, self._mu, self._mat.get('vertexIds')[0].tolist(), self._mat.get('faceIndices').tolist())
        self._mean_mesh.VIDYA_PCA_Data.mat_file_path = f'{self._mat_path.resolve()}'
        self._mean_mesh.VIDYA_PCA_Data.mat_file_name = self._mat_path.stem

        self._mean_mesh.VIDYA_PCA_Features.mat_file_path = f'{self._mat_path.resolve()}'
        self._mean_mesh.VIDYA_PCA_Features.mat_file_name = self._mat_path.stem

        self._mean_mesh.VIDYA_PCA_Data.createSliders()
        self._mean_mesh.VIDYA_PCA_Features.createSliders()

        return {'FINISHED'}