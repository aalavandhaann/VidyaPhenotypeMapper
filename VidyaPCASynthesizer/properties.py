import bpy

class VidyaPCAEigenValues(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Linked Library Object", description="Filename (or mesh object name)")
    blendfilepath: bpy.props.StringProperty(name="Library Blend file path", description="Path to the linked library Blend file")
    librarytypepath: bpy.props.StringProperty(name="Path to type of library item", description="The final path to access all items inside this type of linked library type (eg Object, Armature, Camera etc)")
    filepath: bpy.props.StringProperty(name="Final path to library item", description="The final path to access this mesh object from the linked library Blend file and the library type")
    loaditem: bpy.props.BoolProperty(name="Load Library Item", description="Load the library item into this occlusion creator scene", default=False, update=addLibraryItem)

def register():
    bpy.types.Scene.VIDYA_PCA_Matrix = bpy.props.StringProperty(name="Library Blend file path", default="./mat.mat", description="Path to the linked library Blend file", subtype="FILE_PATH")

def unregister():
    bpy.props.RemoveProperty(bpy.types.Scene, attr='VIDYA_PCA_Matrix')