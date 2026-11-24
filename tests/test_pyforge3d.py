import math
import os
import sys
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pygame
from pyforge3d import (
    AABB3D,
    Camera3D,
    DirectionalLight,
    FirstPersonController,
    Fog3D,
    Matrix4,
    Mesh3D,
    OBJLoader,
    OrbitCamera,
    PointLight,
    Primitives,
    Renderer3D,
    Scene3D,
    Transform3D,
    VoxelPhysicsBody,
    VoxelMaterial,
    VoxelWorld,
    clip_polygon_near_plane,
    compute_face_normals,
)


class TestPyForge3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        cls.screen = pygame.display.set_mode((800, 600))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_01_transforms_and_near_plane_clipping(self) -> None:
        t = Transform3D(x=2.0, y=3.0, z=4.0, rotation_y=math.pi * 0.5, scale_x=2.0, scale_y=2.0, scale_z=2.0)
        verts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        world_v = t.apply_to_vertices(verts)
        self.assertEqual(world_v.shape, (1, 3))
        self.assertAlmostEqual(float(world_v[0, 1]), 3.0, places=4)

        # Polygon straddling near plane z = 0.2
        straddling = np.array(
            [
                [-1.0, -1.0, -0.5],
                [-1.0, +1.0, +1.0],
                [+1.0, +1.0, +1.0],
                [+1.0, -1.0, -0.5],
            ],
            dtype=np.float32,
        )
        clipped = clip_polygon_near_plane(straddling, near_z=0.2)
        self.assertGreaterEqual(len(clipped), 3)
        self.assertTrue(np.all(clipped[:, 2] >= 0.1999))

    def test_02_camera_and_orbit_controllers(self) -> None:
        cam = Camera3D(position=(0.0, 2.0, -5.0), fov=75.0)
        cam.look_at((0.0, 2.0, 0.0))
        self.assertAlmostEqual(cam.yaw, 0.0, places=4)
        self.assertAlmostEqual(cam.pitch, 0.0, places=4)

        pts_w = np.array([[0.0, 2.0, 0.0], [1.0, 3.0, 2.0]], dtype=np.float32)
        pts_c = cam.world_to_camera(pts_w)
        self.assertAlmostEqual(float(pts_c[0, 2]), 5.0, places=4)
        pts_2d = cam.project_camera_vertices(pts_c, 800, 600)
        self.assertAlmostEqual(float(pts_2d[0, 0]), 400.0, places=2)
        self.assertAlmostEqual(float(pts_2d[0, 1]), 300.0, places=2)

        orbit = OrbitCamera(cam, target=(0.0, 0.0, 0.0), distance=10.0, azimuth=0.0, elevation=0.0)
        self.assertAlmostEqual(float(np.linalg.norm(cam.position)), 10.0, places=3)
        orbit.orbit(delta_azimuth=0.5, delta_elevation=0.2, delta_distance=-2.0)
        self.assertAlmostEqual(float(np.linalg.norm(cam.position)), 8.0, places=3)

        fps = FirstPersonController(cam, move_speed=5.0)
        fps.handle_mouse_motion(20, -10)

    def test_03_primitives_and_obj_loader(self) -> None:
        cube = Primitives.cube(size=2.0, position=(0.0, 0.0, 5.0))
        pyr = Primitives.pyramid(base_size=1.5, height=2.0)
        sph = Primitives.sphere(radius=1.0, rings=6, sectors=8)
        cyl = Primitives.cylinder(radius=0.5, height=1.5, segments=8)
        pln = Primitives.plane(width=4.0, depth=4.0)

        self.assertEqual(len(cube.faces), 6)
        self.assertEqual(len(pyr.faces), 5)
        self.assertGreater(len(sph.faces), 20)
        self.assertEqual(len(cyl.faces), 10)
        self.assertEqual(len(pln.faces), 4)

        # Verify all 6 cube face normals point outward from center
        verts_w = cube.get_world_vertices()
        center = verts_w.mean(axis=0)
        for face in cube.faces:
            fv = verts_w[list(face)]
            n = compute_face_normals(fv[0], fv[1], fv[2])
            face_center = fv.mean(axis=0)
            self.assertGreater(float(np.dot(n, face_center - center)), 0.0)

        obj_str = """
        # Simple tetrahedron OBJ
        v 0.0 0.0 0.0
        v 1.0 0.0 0.0
        v 0.5 1.0 0.0
        v 0.5 0.5 1.0
        f 1 3 2
        f 1 2 4
        f 2 3 4
        f 3 1 4
        """
        obj_mesh = Mesh3D.from_obj(obj_str, color=(255, 199, 51))
        self.assertEqual(len(obj_mesh.vertices), 4)
        self.assertEqual(len(obj_mesh.faces), 4)

    def test_04_voxel_world_neighbor_face_culling_and_raycast(self) -> None:
        world = VoxelWorld()
        # A single isolated block has 6 exposed faces
        world.set_block(0, 0, 0, color=(56, 189, 248))
        quads, normals, colors = world.get_culled_quads()
        self.assertEqual(len(quads), 6)

        # Placing an adjacent block at (1, 0, 0) must cull the 2 shared internal faces (6 + 6 - 2 = 10 faces)
        world.set_block(1, 0, 0, color=(168, 85, 247))
        quads2, _, _ = world.get_culled_quads()
        self.assertEqual(len(quads2), 10)

        # A 3x3x3 solid cube of 27 blocks has 27*6 = 162 total faces, but only 9*6 = 54 external faces!
        world.clear()
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    world.set_block(x, y, z)
        quads_3x3, _, _ = world.get_culled_quads()
        self.assertEqual(len(quads_3x3), 54)

        # Raycast from (1.5, 1.5, -3.0) looking toward +Z (0, 0, 1) must hit block (1, 1, 0) on its -Z face
        hit = world.raycast(origin=(1.5, 1.5, -3.0), direction=(0.0, 0.0, 1.0), max_distance=10.0)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.block_pos, (1, 1, 0))
        self.assertEqual(hit.face_normal, (0, 0, -1))
        self.assertEqual(hit.adjacent_pos, (1, 1, -1))

    def test_05_aabb_physics_and_voxel_collisions(self) -> None:
        world = VoxelWorld()
        world.generate_flat_terrain(width=6, depth=6, y_level=0)
        body = VoxelPhysicsBody(position=(2.5, 2.2, 2.5), height=1.6)

        # Simulate falling onto the y=0 voxel floor (top surface at y=1.0)
        for _ in range(45):
            body.step(world, dt=1.0 / 60.0)
        self.assertTrue(body.on_ground)
        self.assertAlmostEqual(body.aabb.min_y, 1.0, delta=0.08)

        # Jump from the ground
        self.assertTrue(body.jump())
        body.step(world, dt=1.0 / 60.0)
        self.assertFalse(body.on_ground)
        self.assertGreater(body.vy, 0.0)

    def test_06_renderer_lighting_fog_and_scene(self) -> None:
        cam = Camera3D(position=(4.0, 5.0, -5.0), yaw=0.0, pitch=-0.35, fov=70.0)
        light = DirectionalLight(direction=(-0.4, -1.0, -0.3), ambient=0.35)
        pt_light = PointLight(position=(4.0, 8.0, 4.0))
        fog = Fog3D(color=(14, 20, 36), near=8.0, far=28.0)

        renderer = Renderer3D(width=800, height=600, light=light, fog=fog)
        world = VoxelWorld()
        world.generate_hills_terrain(width=6, depth=6, max_height=2)

        scene = Scene3D()
        cube = Primitives.cube(size=1.2, color=(255, 199, 51), position=(3.0, 4.0, 3.0), outline_color=(255, 255, 255))
        wire_sphere = Primitives.sphere(radius=0.8, rings=5, sectors=8, position=(1.5, 4.0, 2.5))
        wire_sphere.wireframe = True
        scene.add_mesh(cube, wire_sphere)

        hit = world.raycast(cam.position, cam.forward, max_distance=15.0)
        faces_drawn = renderer.render_scene(self.screen, cam, scene=scene, voxel_world=world, highlight_hit=hit)
        self.assertGreater(faces_drawn, 10)

        # Also test with PointLight
        renderer.light = pt_light
        faces_drawn_pt = renderer.render_scene(self.screen, cam, scene=scene, voxel_world=world)
        self.assertEqual(faces_drawn_pt, faces_drawn)

    def test_07_voxel_material_trees_sky_and_projection(self) -> None:
        world = VoxelWorld()
        grass = VoxelMaterial(top=(56, 189, 78), side=(130, 92, 58), bottom=(105, 72, 42))
        world.set_block(2, 0, 2, grass)
        world.place_tree(2, 1, 2, height=3)
        self.assertGreaterEqual(world.get_highest_solid_y(2, 2), 4)

        cam = Camera3D(position=(2.5, 4.0, -4.0), yaw=0.0, pitch=-0.2)
        proj = cam.project_world_point((2.5, 1.5, 2.5), 800, 600)
        self.assertIsNotNone(proj)
        assert proj is not None
        self.assertGreater(proj[2], 0.5)

        body = VoxelPhysicsBody(position=(2.5, 2.0, 2.5))
        self.assertTrue(body.intersects_block(2, 2, 2))
        self.assertFalse(body.intersects_block(5, 5, 5))

        renderer = Renderer3D(800, 600)
        renderer.draw_sky_gradient(self.screen, cam)
        drawn = renderer.render_scene(self.screen, cam, voxel_world=world)
        self.assertGreater(drawn, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
