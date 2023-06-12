import bpy
import pathlib

import numpy as np
import scipy.io as sio


if __name__ == '__main__':
    mat_topology_path: pathlib.Path = pathlib.Path(bpy.path.abspath('//')).joinpath('matrices').joinpath('thorax_topology.mat')
    C: bpy.types.Context = bpy.context
    mesh: bpy.types.Object = C.view_layer.objects.get('Human')
    mesh_vertices = mesh.data.vertices
    mesh_polygons = mesh.data.polygons
    mesh_loops = mesh.data.loops

    vertices: list[int] = []
    faces: list[list[int]] = []

    for v in mesh.data.vertices:
        if(v.select):
            vertices.append(v.index)

    for p in mesh.data.polygons:
        if(p.select):
            polygon_vertices: list[int] = []
            for lid in p.loop_indices:
                loop = mesh_loops[lid]
                polygon_vertices.append(loop.vertex_index)
                    
            faces.append(polygon_vertices)

    vertices_np = np.array(vertices)
    faces_np = np.array(faces)

    sio.savemat(f'{mat_topology_path.resolve()}', {'Thorax.vertexIds': vertices_np, 'Thorax.faceIndices': faces_np})