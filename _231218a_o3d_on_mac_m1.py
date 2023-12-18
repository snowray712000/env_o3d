""" 成功在 mac 也能使用 open3d
- 使用 miniconda
- 使用 python 3.10 版
"""
#%%
import numpy as np
import open3d as o3d
from Easy import TpO3d


app = TpO3d.Application()
app.initialize()

win3D = app.create_window("3D",858,480)

widgetScene = TpO3d.SceneWidget()
win3D.add_child(widgetScene)
scene3d = TpO3d.Open3DScene(win3D.renderer)
widgetScene.scene = scene3d

bunny = TpO3d.BunnyMesh()
mesh = TpO3d.IoModule().read_triangle_mesh(bunny.path)
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.compute_vertex_normals()

mat = TpO3d.MaterialRecord()
mat.shader = "defaultLit"
scene3d.add_geometry("mesh", mesh, material=mat, add_downsampled_copy_for_fast_rendering=True)

def fn_mouse(event_mouse: TpO3d.MouseEvent)->TpO3d.Widget.EventCallbackResult:
    if event_mouse.type == TpO3d.MouseEvent.Type.BUTTON_DOWN and event_mouse.is_button_down(TpO3d.MouseButton.RIGHT):
        def fn_img(depth_image: TpO3d.Image):
            img = np.array(depth_image)
            u = event_mouse.x - widgetScene.frame.x
            v = event_mouse.y - widgetScene.frame.y
            z = img[v][u]
            
            if z == 1.0: # 沒找到
                print('no click anything u', u, ' v', v)
            else:                
                pt = scene3d.camera.unproject(u, v, z, widgetScene.frame.width, widgetScene.frame.height)
                print(f'u {u} v {v} z {z} pt:{pt[0]} {pt[1]} {pt[2]}')
            pass            
        scene3d.scene.render_to_depth_image(fn_img)        
        return TpO3d.Widget.EventCallbackResult.CONSUMED
    return TpO3d.Widget.EventCallbackResult.IGNORED

widgetScene.set_on_mouse(fn_mouse)

app.run()