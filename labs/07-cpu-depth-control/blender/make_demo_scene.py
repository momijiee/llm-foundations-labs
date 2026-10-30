"""Run with Blender -b -P; scene follows the instructor's appendix A.1."""
import argparse
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return mat


def add_area_light(name, location, energy, size, target):
    data = bpy.data.lights.new(name, type='AREA')
    data.energy, data.size = energy, size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new('World')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0.15, 0.15, 0.15, 1)
    body = material('White matte', (0.8, 0.8, 0.8), 0.7)
    floor_mat = material('Floor', (0.65, 0.65, 0.65), 0.8)
    bpy.ops.mesh.primitive_plane_add(size=24)
    bpy.context.object.name = 'Floor'
    bpy.context.object.data.materials.append(floor_mat)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.62, segments=64, ring_count=32,
                                        location=(-0.95, 0, 0.62))
    sphere = bpy.context.object
    sphere.name = 'Sphere'
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_cube_add(size=1.05, location=(0.55, 0.15, 0.525))
    cube = bpy.context.object
    cube.name = 'BeveledCube'
    bevel = cube.modifiers.new('Bevel', 'BEVEL')
    bevel.width, bevel.segments = 0.09, 4
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.37, depth=1.25,
                                       location=(1.6, 0.65, 0.625))
    bpy.context.object.name = 'Cylinder'
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_torus_add(major_radius=0.44, minor_radius=0.15,
                                    major_segments=64, minor_segments=24,
                                    location=(-0.15, 1.25, 0.59),
                                    rotation=(1.57079632679, 0, 0))
    bpy.context.object.name = 'Torus'
    bpy.ops.object.shade_smooth()
    for obj in scene.objects:
        if obj.type == 'MESH' and obj.name != 'Floor':
            obj.data.materials.append(body)
    camera_data = bpy.data.cameras.new('Camera')
    camera_data.lens = 63
    camera = bpy.data.objects.new('Camera', camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (4.8, -6, 2.8)
    look_at(camera, (0, 0, 0.55))
    scene.camera = camera
    add_area_light('Key', (3, -3.2, 4.2), 300, 3.5, (0, 0, 0.5))
    add_area_light('Fill', (-3.6, -2.8, 2.2), 80, 5, (0, 0, 0.6))
    add_area_light('Rim', (0.5, 3.5, 3.2), 220, 2.2, (0, 0, 0.8))
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))


if __name__ == '__main__':
    main()
