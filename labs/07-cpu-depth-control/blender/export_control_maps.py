"""Export Mist/Normal passes and a white material override in one CPU render."""
import argparse
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view


def scene_depth_range(scene, camera):
    # View-frustum filtering follows appendix A.2; it is not an occlusion test.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    distances = []
    for obj in scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            for vertex in mesh.vertices:
                world = evaluated.matrix_world @ vertex.co
                ndc = world_to_camera_view(scene, camera, world)
                if ndc.z > 0 and 0 <= ndc.x <= 1 and 0 <= ndc.y <= 1:
                    distances.append((world - camera.matrix_world.translation).length)
        finally:
            evaluated.to_mesh_clear()
    if not distances:
        raise RuntimeError('No mesh vertices inside camera frustum; check the camera.')
    near, far = np.percentile(distances, [2, 98]).astype(float)
    return near, far if far > near else near + 1


def make_clay_material():
    mat = bpy.data.materials.new('Clay override')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)
    bsdf.inputs['Roughness'].default_value = 0.7
    return mat


def setup_compositor(scene, outdir, name):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    render = tree.nodes.new('CompositorNodeRLayers')
    invert = tree.nodes.new('CompositorNodeInvert')
    invert.inputs['Fac'].default_value = 1
    tree.links.new(render.outputs['Mist'], invert.inputs['Color'])
    multiply = tree.nodes.new('CompositorNodeMixRGB')
    multiply.blend_type = 'MULTIPLY'
    multiply.inputs[0].default_value = 1
    multiply.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    add = tree.nodes.new('CompositorNodeMixRGB')
    add.blend_type = 'ADD'
    add.inputs[0].default_value = 1
    add.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    tree.links.new(render.outputs['Normal'], multiply.inputs[1])
    tree.links.new(multiply.outputs[0], add.inputs[1])
    for suffix, socket in [('depth', invert.outputs[0]), ('normal', add.outputs[0])]:
        output = tree.nodes.new('CompositorNodeOutputFile')
        output.base_path = str(outdir)
        output.file_slots[0].path = f'{name}_{suffix}_'
        output.format.file_format = 'PNG'
        output.format.color_mode = 'RGB'
        output.format.color_depth = '8'
        # Raw override is required even for PNG: save_as_render=False alone
        # still applies the output sRGB transfer (linear 0.5 becomes ~188).
        output.format.color_management = 'OVERRIDE'
        output.format.view_settings.view_transform = 'Raw'
        output.format.view_settings.look = 'None'
        output.format.view_settings.exposure = 0
        output.format.view_settings.gamma = 1
        output.save_as_render = True
        tree.links.new(socket, output.inputs[0])
    composite = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(render.outputs['Image'], composite.inputs['Image'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--name', default='demo')
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--threads', type=int, default=12)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = args.threads
    scene.render.resolution_x = scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.frame_set(1)
    layer = bpy.context.view_layer
    layer.use_pass_mist = layer.use_pass_normal = True
    layer.material_override = make_clay_material()
    near, far = scene_depth_range(scene, scene.camera)
    scene.world.mist_settings.start = near
    scene.world.mist_settings.depth = far - near
    scene.world.mist_settings.falloff = 'LINEAR'
    setup_compositor(scene, outdir, args.name)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.filepath = str(outdir / f'{args.name}_clay.png')
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - start
    record = dict(resolution=args.res, samples=args.samples, threads=args.threads,
                  near=near, far=far, render_and_export_s=elapsed,
                  timing_note='clay/depth/normal share one Cycles render; no separate render times',
                  blender_version=bpy.app.version_string)
    for suffix in ['clay', 'depth_0001', 'normal_0001']:
        if not (outdir / f'{args.name}_{suffix}.png').is_file():
            raise RuntimeError(f'Missing export: {suffix}')
    (outdir / f'{args.name}_controls.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('CONTROL_EXPORT_RECORD=' + json.dumps(record))


if __name__ == '__main__':
    main()
