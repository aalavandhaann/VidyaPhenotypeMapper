import bpy

from VidyaPCASynthesizer.operators import VidyaPCASynthesizer

class VIDYA_PCA_Slider_Items(bpy.types.UIList):
    bl_idname = 'VIEW3D_UL_VIDYA_PCASliderItems'

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, active_index):
        grid = layout.grid_flow(row_major=True, columns=1, align=True)
        row = grid.row()
        row.prop(item, 'coefficient', text='')

class PT_1_VidyaPCAMatrixLoader(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_vidya_pca_matrix_loader_panel"
    bl_label = "Load PCA Matrix"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VIDYA"
    bl_description = "Load the PCA matrix file created using Thorax-PCA.py inside bl-scripts"

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text = "Select Matrix")
        row = box.row(align=True)
        row.prop(context.scene, "VIDYA_PCA_Matrix")

        

class PT_2_VidyaPCASliders(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_vidya_pca_sliders_panel"
    bl_label = "PCA Sliders"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VIDYA"
    bl_description = "The sliders of the PCA matrix file loaded"

    @classmethod
    def poll(self, context):
        return context.active_object != None and context.active_object.type == 'MESH'

    def draw(self, context):
        mesh: bpy.types.Object = context.active_object
        layout = self.layout
        box = layout.box()
        box.label(text = "PCA Sliders")

        if(not len(mesh.VIDYA_PCA_Data.sliders)):
            box.operator(VidyaPCASynthesizer.bl_idname, icon='FILE_REFRESH', text="")
        else:
            row = box.row(align=True)
            row.template_list(VIDYA_PCA_Slider_Items.bl_idname, "", mesh.VIDYA_PCA_Data,
                          'sliders', mesh.VIDYA_PCA_Data, "slider_index", rows=3, maxrows=3, type="DEFAULT")