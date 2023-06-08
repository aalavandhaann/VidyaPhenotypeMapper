import bpy

def loadMATFile(self, context):
    bpy.ops.vidya.pcasynthesizer('EXEC_DEFAULT')

class VidyaPCAEigenSlider(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Linked Library Object", description="Filename (or mesh object name)")
    value: bpy.props.FloatProperty(name="Slider Value", default=0.0, subtype='FACTOR')

def register():
    bpy.types.Scene.VIDYA_PCA_Matrix = bpy.props.StringProperty(name="Library Blend file path", default="./mat.mat", description="Path to the linked library Blend file", subtype="FILE_PATH", update=loadMATFile)

def unregister():
    bpy.props.RemoveProperty(bpy.types.Scene, attr='VIDYA_PCA_Matrix')