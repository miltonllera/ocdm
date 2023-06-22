import bpy
import os
import math
import colorsys
import itertools
import csv
import argparse


# =========================
# PARAMETER GENERATION
# =========================
def linspace(start, end, num):
    if num == 1:
        return [start]
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]


def generate_colors(n):
    # Simple RGB sampling (can be improved later)
    hues = linspace(0.0, 1.0, n)
    colors = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues]
    return colors


# =========================
# HELPER FUNCTIONS
# =========================
def set_object_to_origin(obj):
    obj.location = (0, 0, 1)  # but raised slightly above the ground


def set_scale(obj, s):
    obj.scale = (s, s, s)


def set_rotation(obj, angle):
    obj.rotation_euler = (0, 0, angle)


def set_color(obj, rgba):
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="Material")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]

    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba


def rotate_camera_around_origin(cam, angle, base_radius, base_z):
    cam.location.x = base_radius * math.cos(angle)
    cam.location.y = base_radius * math.sin(angle)
    cam.location.z = base_z

    direction = -cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()


def reset_object(obj, original_state):
    obj.location = original_state["location"]
    obj.rotation_euler = original_state["rotation"]
    obj.scale = original_state["scale"]


# =========================
# MAIN
# =========================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "metadata.csv")

    # Get collection
    collections = bpy.data.collections  # type: ignore
    camera = bpy.data.objects[args.camera]  # type: ignore
    pentominos = [obj for obj in collections['Pentominos'].objects if obj.type == 'MESH']
    wall_and_floor = [obj for obj in collections['Background'].objects if obj.type == 'MESH']

    # Generate parameter grids
    scales = linspace(args.scale_min, args.scale_max, args.scale_steps)
    rotations = linspace(args.rot_min, args.rot_max, args.rot_steps)
    camera_angles = linspace(args.cam_min, args.cam_max, args.cam_steps)
    colors = generate_colors(args.color_steps)

    # Store original states
    original_states = {}
    for obj in pentominos:
        original_states[obj.name] = {
            "location": obj.location.copy(),
            "rotation": obj.rotation_euler.copy(),
            "scale": obj.scale.copy()
        }

    base_radius = camera.location.length
    base_z = camera.location.z

    # CSV setup
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "filename",
            "object",
            "scale",
            "rotation_z",
            "color_r", "color_g", "color_b", "color_a",
            "camera_angle"
        ])

        counter = 0

        for obj in pentominos:
            for scale, rot, color, cam_angle in itertools.product(
                scales, rotations, colors, camera_angles
            ):
                reset_object(obj, original_states[obj.name])

                set_object_to_origin(obj)
                set_scale(obj, scale)
                set_rotation(obj, rot)
                set_color(obj, color)

                rotate_camera_around_origin(
                    camera, cam_angle, base_radius, base_z
                )

                filename = f"{obj.name}_{counter:05d}.png"
                filepath = os.path.join(args.output_dir, filename)
                bpy.context.scene.render.filepath = filepath

                bpy.ops.render.render(write_still=True)

                writer.writerow([
                    filename,
                    obj.name,
                    scale,
                    rot,
                    color[0], color[1], color[2], color[3],
                    cam_angle
                ])

                counter += 1

    print(f"Done! Rendered {counter} images.")
    print(f"Metadata saved to: {csv_path}")


# =========================
# ARGPARSE ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender dataset generator")

    # Paths / scene
    parser.add_argument("--output_dir", type=str, default="data/temp/pentominos_3d")
    parser.add_argument("--camera", type=str, default="Camera")

    # Scale
    parser.add_argument("--scale_min", type=float, default=0.5)
    parser.add_argument("--scale_max", type=float, default=1.5)
    parser.add_argument("--scale_steps", type=int, default=3)

    # Object rotation
    parser.add_argument("--rot_min", type=float, default=0.0)
    parser.add_argument("--rot_max", type=float, default=math.pi)
    parser.add_argument("--rot_steps", type=int, default=3)

    # Camera rotation
    parser.add_argument("--cam_min", type=float, default=-math.pi / 4)
    parser.add_argument("--cam_max", type=float, default=math.pi / 4)
    parser.add_argument("--cam_steps", type=int, default=3)

    # Color resolution
    parser.add_argument("--color_steps", type=int, default=2,
                        help="Number of steps per RGB channel")

    args = parser.parse_args()

    main(args)
