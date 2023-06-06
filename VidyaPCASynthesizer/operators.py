class VidyaPCASynthesizer(bpy.types.Operator):
    bl_idname = 'vidya.pcasynthesizer'
    bl_label = 'PCA Synthesizer'
    bl_description = "Given the matrix file start synthesizing different shapes"
    bl_options = {'REGISTER', 'UNDO'}