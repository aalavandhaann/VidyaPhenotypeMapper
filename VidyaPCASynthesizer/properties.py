import bpy



def register():
    bpy.types.Scene.VIDYA_PCA_Matrix = bpy.props.StringProperty(name="Library Blend file path", default="./mat.mat", description="Path to the linked library Blend file", subtype="FILE_PATH")

def unregister():
    bpy.props.RemoveProperty(bpy.types.Scene, attr='VIDYA_PCA_Matrix')