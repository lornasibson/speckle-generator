import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import pyvale
import mooseherder as mh
import bpy

def main() -> None:
    # data_path = pyvale.DataSet.render_mechanical_3d_path()
    # sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    # disp_comps = ("disp_x","disp_y", "disp_z")

    # # Scale m -> mm
    # # NOTE: All lengths are to be specified in mm
    # sim_data = pyvale.scale_length_units(sim_data,disp_comps,1000.0)

    # render_mesh = pyvale.create_render_mesh(sim_data,
    #                                     ("disp_y","disp_x"),
    #                                     sim_spat_dim=3,
    #                                     field_disp_keys=disp_comps)

    # TODO: Get sample sim - add here

    # Set the save path
    # --------------------------------------------------------------------------
    base_dir = Path.cwd() / "RBM/pipe_reflections/out_of_plane"

    # Creating the scene
    # --------------------------------------------------------------------------
    scene = pyvale.BlenderScene()

    # Add circle sample
    bpy.ops.mesh.primitive_circle_add(radius=25.0, fill_type='NGON',
                                             align='WORLD', location=(0.0, 0.0, 0.0),
                                             rotation=(0.0, 0.0, 0.0))
    circle = bpy.context.view_layer.objects.active
    circle.name = "Sample"

    # Add the camera
    cam_data_0 = pyvale.CameraData(pixels_num=np.array([2464, 2056]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=np.array([-25, 0, 1610]), # TODO: Work out what to make this value
                                 rot_world=Rotation.from_euler("xyz", [0, -2.5, 0], degrees=True),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=75.0,
                                 fstop=2.8)
    cam_data_1 = pyvale.CameraData(pixels_num=np.array([2464, 2056]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=np.array([25, 0, 1610]), # TODO: Work out what to make this value
                                 rot_world=Rotation.from_euler("xyz", [0, 2.5, 0], degrees=True),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=75.0,
                                 fstop=2.8)

    stereo_system = pyvale.CameraStereo(cam_data_0, cam_data_1)

    stereo_system.save_calibration_mid(base_dir)

    scene.add_stereo_system(stereo_system)


    for cam in [obj for obj in bpy.data.objects if obj.type == "CAMERA"]:
        bpy.context.scene.camera = cam
        cam.data.clip_end = 3500.0

    bpy.context.scene.camera.data.clip_end = 3500.0

    # Apply the speckle pattern
    material_data = pyvale.BlenderMaterialData()
    speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()

    mm_px_resolution = pyvale.CameraTools.calculate_mm_px_resolution(cam_data_0)
    scene.add_speckle(part=circle,
                      speckle_path=speckle_path,
                      mat_data=material_data,
                      mm_px_resolution=mm_px_resolution)

    # Add pipe
    bpy.ops.mesh.primitive_cylinder_add(radius=38.0, depth=1333.0,
                                                    end_fill_type="NOTHING",
                                                    align='WORLD',
                                                    location=(0.0, 0.0, 666.5),
                                                    rotation=(0.0, 0.0, 0.0))
    pipe = bpy.context.view_layer.objects.active
    pipe.name = "Pipe"
    pyvale.BlenderTools.clear_material_nodes(pipe)
    node_tree = bpy.data.materials["Material.001"].node_tree
    mat_nodes = bpy.data.materials["Material.001"].node_tree.nodes
    bsdf = mat_nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Metallic"].default_value = 0.7
    output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Add ring light
    bpy.ops.mesh.primitive_torus_add(align="WORLD",
                                                  location=(0.0, 0.0, 1340),
                                                  rotation=(0.0, 0.0, 0.0),
                                                  major_segments=100,
                                                  minor_segments=25,
                                                  mode="EXT_INT",
                                                  abso_major_rad=58.0,
                                                  abso_minor_rad=33.0)
    ringlight = bpy.context.view_layer.objects.active
    ringlight.name = "RingLight"
    pyvale.BlenderTools.clear_material_nodes(ringlight)
    node_tree = bpy.data.materials["Material.002"].node_tree
    mat_nodes = bpy.data.materials["Material.002"].node_tree.nodes
    emission = mat_nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 804.519492552*2 # W/m2
    output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    # light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
    #                                      pos_world=(0, 0, 1400),
    #                                      rot_world=Rotation.from_euler("xyz",
    #                                                                    [0, 0, 0]),
    #                                      energy=23)
    # light = scene.add_light(light_data)



    # Rendering image
    # --------------------------------------------------------------------------
    # Set this to True to render image of the current scene
    render_opts = True
    if render_opts:
        # NOTE: If no save directory is specified, this is where the images will
        # be saved
        render_data = pyvale.RenderData(cam_data=(stereo_system.cam_data_0,
                                                  stereo_system.cam_data_1),
                                        base_dir=base_dir,
                                        threads=256,
                                        samples=15)
        # NOTE: The number of threads used to render the images is set within
        # RenderData, it is defaulted to 4 threads

        scene.render_single_image(stage_image=False,
                                  render_data=render_data)
        for i in range(2):
            path_name = base_dir / ("blenderimages/blenderimage_0_" + str(i) + ".tiff")
            new_path = path_name.with_name(("blender_image_0_0_" + str(i) + ".tiff"))
            path_name.replace(new_path)

        # NOTE: If bounce_image is set to True, the image will be saved to disk,
        # converted to an array, deleted and the image array will be returned.
        for i in range(10):
            added_disp = 0.1
            circle.location[2] += added_disp
            print(f"{circle.location=}")
            scene.render_single_image(stage_image=False,
                                      render_data=render_data)

            for j in range(2):
                path_name = base_dir / ("blenderimages/blenderimage_0_" + str(j) + ".tiff")
                new_path = path_name.with_name(("blender_image_" + str(i) + "_" + str(j) + ".tiff"))
                path_name.replace(new_path)

        print()
        print(80*"-")
        print("Save directory of the image:", (render_data.base_dir / "blenderimages"))
        print(80*"-")
        print()

    # Save Blender file
    # --------------------------------------------------------------------------
    # The file that will be saved is a Blender project file. This can be opened
    # with the Blender GUI to view the scene.

if __name__ == "__main__":
    main()
