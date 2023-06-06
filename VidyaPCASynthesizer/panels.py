import bpy


class PT_1_VidyaOcclusionsLibraryLoader(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_vidya_occlusions_libraryloader_panel"
    bl_label = "Load Occlusions Library"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VIDYA"


    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text = "Select Matrix")
        row = box.row(align=True)
        row.prop(context.scene, "VIDYA_PCA_Matrix")
