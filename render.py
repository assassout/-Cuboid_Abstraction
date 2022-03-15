import os
import re

import bpy
import bmesh
import mathutils
import numpy as np
import time

# workflow : load obj => create obj material => set up light property => set up camera property=> => set up scene nodes => set up camera positon => render


############### clear blender ###################
#clear objects in scene
def clean():
    for obj in bpy.data.objects:
        if obj.name in [
                "Camera",
                "Light",
        ]:
            continue
        bpy.data.objects.remove(obj)

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


########### util's function #####################
def load_obj_model(path):  #if load sucess return ture else false
    ext = os.path.splitext(path)[1]
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=path)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise RuntimeError("import failed")
    model_name = os.path.basename(path)
    model_name_no_ext = os.path.splitext(model_name)[0]
    bpy.data.objects[model_name_no_ext].name = model_name
    obj = bpy.data.objects[model_name]
    obj.rotation_euler = (np.pi * 0.5, 0, 0)  # the unit is radians
    bpy.context.view_layer.update()
    return obj


class Render(object):
    """
    render mesh
    """
    def __init__(self, color_list, alpha_list, frame_num) -> None:
        self.scene = bpy.data.scenes["Scene"]
        self.cam = bpy.data.cameras["Camera"]
        self.cam_pos_list = []
        self.frame_num = frame_num
        self.setup_random_cam_pos(360 / self.frame_num, self.frame_num)
        self.lights = []
        self.init_light()
        self.materials = []
        self.init_material(color_list, alpha_list)
        self.objs = []

    def init_material(self, color_list, alpha_list):
        for i in range(len(color_list)):
            mat = bpy.data.materials.new(name=str(i))
            mat.use_nodes = True
            mat_nodes = mat.node_tree.nodes
            bsdf = mat_nodes['Principled BSDF']
            bsdf.inputs["Base Color"].default_value = color_list[i]
            bsdf.inputs["Alpha"].default_value = alpha_list[i]
            self.materials.append(mat)

    def setup_camera(self):
        pass

    def mass_center(self):
        bb_pts = np.array([])
        for obj in self.objs:
            matrix_world = obj.matrix_world.copy()
            bb_world = [
                matrix_world @ mathutils.Vector(v) for v in obj.bound_box
            ]
            bb_pts = np.append(bb_pts, bb_world)
        bb_pts = np.reshape(bb_pts, (-1, 3))
        bb_max_pt = np.max(bb_pts, axis=0)
        bb_min_pt = np.min(bb_pts, axis=0)
        center = (bb_max_pt + bb_min_pt) / 2
        return center

    def setup_camera_view(self, cam_pos, cam_end):
        cam_pos = mathutils.Vector(cam_pos)
        cam_end = mathutils.Vector(cam_end)
        cam = bpy.data.objects['Camera']
        cam.location = cam_pos
        direction = cam_end - cam_pos
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        # assume we're using euler rotation
        cam.rotation_euler = rot_quat.to_euler()

    def setup_random_cam_pos(self, delta_angle, num):
        cam_pos = bpy.data.objects['Camera'].location
        for i in range(num):
            angel = delta_angle * i
            eul = mathutils.Euler((0, 0, np.radians(angel)), 'XYZ')
            mat_rot = eul.to_matrix()
            self.cam_pos_list.append(mat_rot @ cam_pos)

    def add_light(self, name, type):
        view_layer = self.scene.view_layers[0]
        light = bpy.data.lights.new(name=name, type=type)
        light_obj = bpy.data.objects.new(name=name, object_data=light)
        view_layer.active_layer_collection.collection.objects.link(light_obj)
        return light, light_obj

    def init_light(self):
        light_0 = bpy.data.lights["Light"]
        self.lights.append(light_0)
        bpy.data.objects["Light"].hide_render = True

        light_1, light_obj_1 = self.add_light(name="Light_1", type="AREA")
        light_1.energy = 50000
        light_1.shape = 'SQUARE'
        light_1.size = 100
        light_obj_1.rotation_euler = np.array([0, 0, 0])
        light_obj_1.location = bpy.data.objects[
            "Light"].location * mathutils.Vector(np.array([1, 1, 1]))
        self.lights.append(light_1)

        light_2, light_obj_2 = self.add_light(name="Light_2", type="AREA")
        light_2.energy = 50000
        light_2.shape = 'SQUARE'
        light_2.size = 100
        light_obj_2.rotation_euler = np.array([np.radians(180), 0, 0])
        light_obj_2.location = bpy.data.objects[
            "Light"].location * mathutils.Vector(np.array([1, 1, -1]))
        self.lights.append(light_2)

    def setup_light(self):
        pass

    def setup_material(self, obj, material):
        """
        docstring
        """
        # obj.data.materials.append(material)
        if len(obj.material_slots) == 0:
            bpy.ops.object.material_slot_add()
        obj.material_slots[0].link = 'OBJECT'
        obj.material_slots[0].material = material

    def setup_shadow(self):
        """
        docstring
        """
        pass

    def setup_scene(self):
        """
        docstring
        """
        self.setup_light()
        self.setup_camera()

        # scene=bpy.context.scene
        self.scene.render.engine = 'CYCLES'
        self.scene.render.resolution_x = 1080
        self.scene.render.resolution_y = 1080
        self.scene.cycles.device = 'GPU'
        self.scene.render.film_transparent = True
        self.scene.frame_start = 1
        self.scene.frame_end = 1

        # backgrand
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
            0].default_value = (1, 1, 1, 1)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
            1].default_value = 1

        # compositor nodes
        self.scene.use_nodes = True
        for node in self.scene.node_tree.nodes:
            self.scene.node_tree.nodes.remove(node)

        tree = self.scene.node_tree
        rl = tree.nodes.new('CompositorNodeRLayers')
        output_node = tree.nodes.new('CompositorNodeOutputFile')
        output_node.format.file_format = 'PNG'
        output_node.format.color_mode = 'RGBA'
        tree.links.new(rl.outputs['Image'], output_node.inputs['Image'])

    def render(self, obj_path_list, save_dir, render_img_name, materials):
        for i in range(len(obj_path_list)):
            self.objs.append(load_obj_model(obj_path_list[i]))
            self.setup_material(self.objs[i], self.materials[materials[i]])

        mass_center = self.mass_center()

        self.setup_scene()
        for i in range(self.frame_num):
            self.scene.frame_set(i)  # it will effect "#.png"
            cam_pos = self.cam_pos_list[i]
            cam_end = mass_center
            self.setup_camera_view(cam_pos, cam_end)
            output_node = self.scene.node_tree.nodes["File Output"]
            output_node.base_path = save_dir
            output_node.file_slots[0].path = "%s_#.png" % (render_img_name)
            bpy.ops.render.render(write_still=True)

    def clean_obj(self):
        for obj in self.objs:
            bpy.data.objects.remove(obj)
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        self.objs.clear()


def render(scene_type, frame_num, radius):
    save_dir = os.path.join(
        os.getcwd(), "img_%.4f/type%d/" % (radius, scene_type))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pc_type = ["input_pc", "GT_pc", "PCN_pc", "MSN_pc", "topnet_pc", "our_pc"]
    # pc_type = ["input_pc", "GT_pc"]

    mesh_dir = "mesh_pc_%.4f/type%d" % (radius, scene_type)
    obj_list = np.loadtxt(os.path.join(mesh_dir, "name.txt"), dtype=np.str)
    # obj_list=np.reshape(obj_list,(-1,len(obj_list)))
    # obj_list=obj_list[0:1]

    color_list = np.array([
        np.array([250, 190, 0, 255]) / 255,  #yellow
        np.array([61, 255, 100, 255]) / 255,  #green
        np.array([1, 0.586, 0.037, 1]),  #yellow
        np.array([0.049, 0.400, 1, 1]),  #blue
    ])
    ############## type3 color
    color_list = np.array([
        np.array([250, 190, 0, 255]) / 255,  #yellow
        np.array([61, 255, 100, 255]) / 255,  #green
        np.array([0.049, 0.400, 1, 1]),  #blue
        np.array([1, 0.586, 0.037, 1]),  #yellow
    ])

    alpha_list = np.array([1, 1, 1, 1, 0.2, 0.2])
    r = Render(color_list, alpha_list, frame_num)
    for obj_path in obj_list:
        obj_path = np.reshape(obj_path, (-1, 2))
        render_img_name = os.path.basename(obj_path[0, 0]).split("_a.ply")[0]

        #####################input_pc
        print(obj_path[0, :])
        r.clean_obj()
        r.render(obj_path[0, :], save_dir, render_img_name + "_" + pc_type[0],
                 [2, 3])

        ##################GT_pc
        print(obj_path[1, :])
        r.clean_obj()
        r.render(obj_path[1, :], save_dir, render_img_name + "_" + pc_type[1],
                 [2, 3])

        ########## PCN_pc
        print(obj_path[2, :])
        r.clean_obj()
        r.render(obj_path[2, :], save_dir, render_img_name + "_" + pc_type[2],
                 [2, 3])

        ############## MSN_pc
        print(obj_path[3, :])
        r.clean_obj()
        r.render(obj_path[3, :], save_dir, render_img_name + "_" + pc_type[3],
                 [2, 3])

        ############## topnet_pc
        print(obj_path[4, :])
        r.clean_obj()
        r.render(obj_path[4, :], save_dir, render_img_name + "_" + pc_type[4],
                 [2, 3])

        ############## our_pc
        print(obj_path[5, :])
        r.clean_obj()
        r.render(obj_path[5, :], save_dir, render_img_name + "_" + pc_type[5],
                 [2, 3])


if __name__ == "__main__":
    # scene_type = [1, 5, 6, 8, 9]
    scene_type = [6,8]
    # scene_type = [1]
    frame_num = 1
    radius = 0.02
    s_t = time.time()
    for i in scene_type:
        clean()
        render(i, frame_num, radius)
    e_t = time.time() - s_t
    print("-----------")
    print(e_t)