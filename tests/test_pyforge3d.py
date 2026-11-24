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
    CANONICAL_QUAD_UV,
    Camera3D,
    ChunkedVoxelWorld,
    DirectionalLight,
    FirstPersonController,
    Fog3D,
    FractalNoise2D,
    Matrix4,
    Mesh3D,
    OBJLoader,
    OrbitCamera,
    PerlinNoise2D,
    PointLight,
    Primitives,
    Renderer3D,
    Scene3D,
    SkyDome3D,
    SoftwareTexturedRasterizer,
    Texture2D,
    Transform3D,
    VoxelPhysicsBody,
    VoxelMaterial,
    VoxelWorld,
    clip_polygon_near_plane,
    clip_textured_polygon_near_plane,
    compute_face_normals,
    hash_2d,
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

    def test_08_procedural_perlin_and_fractal_noise(self) -> None:
        p1 = PerlinNoise2D(seed=42)
        p2 = PerlinNoise2D(seed=42)
        xs = np.linspace(-10.0, 10.0, 32, dtype=np.float32)
        zs = np.linspace(-10.0, 10.0, 32, dtype=np.float32)
        gx, gz = np.meshgrid(xs, zs)

        v1 = p1.sample(gx, gz)
        v2 = p2.sample(gx, gz)
        self.assertEqual(v1.shape, (32, 32))
        self.assertTrue(np.allclose(v1, v2))
        self.assertGreaterEqual(float(v1.min()), -1.05)
        self.assertLessEqual(float(v1.max()), 1.05)

        fbm = FractalNoise2D(seed=99, octaves=4, persistence=0.5, lacunarity=2.0, scale=14.0)
        f_vals = fbm.sample(gx, gz, normalized=True)
        self.assertGreaterEqual(float(f_vals.min()), 0.0)
        self.assertLessEqual(float(f_vals.max()), 1.0)
        # Scalar query matches array query
        self.assertAlmostEqual(
            float(fbm.sample(float(xs[5]), float(zs[7]), normalized=True)),
            float(f_vals[7, 5]),
            places=4,
        )

        h1 = hash_2d(12, -7, seed=1337)
        h2 = hash_2d(12, -7, seed=1337)
        self.assertEqual(h1, h2)
        self.assertGreaterEqual(h1, 0.0)
        self.assertLess(h1, 1.0)

    def test_09_texture2d_and_material_transparency(self) -> None:
        rgb_arr = np.full((16, 16, 3), (60, 180, 90), dtype=np.uint8)
        tex_opaque = Texture2D(rgb_arr)
        self.assertFalse(tex_opaque.has_cutout)
        self.assertEqual(tex_opaque.avg_color, (60, 180, 90))
        surf = tex_opaque.to_surface((32, 32))
        self.assertEqual(surf.get_size(), (32, 32))

        rgba_arr = np.full((16, 16, 4), (180, 220, 255, 110), dtype=np.uint8)
        rgba_arr[0, :, 3] = 255  # opaque border
        tex_glass = Texture2D(rgba_arr)
        self.assertTrue(tex_glass.has_cutout)
        self.assertIsNotNone(tex_glass.alpha)

        from_s = Texture2D.from_surface(surf)
        self.assertEqual(from_s.width, 32)
        self.assertEqual(from_s.height, 32)

        water_mat = VoxelMaterial(
            top=(40, 120, 220),
            side=(35, 110, 210),
            bottom=(30, 95, 195),
            top_texture=tex_opaque,
            alpha=165,
            solid=False,
            name="Water",
        )
        self.assertTrue(water_mat.is_transparent)
        self.assertIs(water_mat.texture_for_normal((0, 1, 0)), tex_opaque)

    def test_10_chunked_voxel_world_streaming_overrides_and_raycast(self) -> None:
        solid_mat = VoxelMaterial(top=(80, 160, 80), side=(120, 90, 60), bottom=(100, 70, 50), solid=True, name="Terrain")
        water_mat = VoxelMaterial(top=(40, 110, 220), side=(40, 110, 220), bottom=(30, 90, 200), alpha=160, solid=False, name="Water")

        def simple_gen(cx: int, cz: int, cs: int, w: ChunkedVoxelWorld) -> None:
            x0, z0 = cx * cs, cz * cs
            for lx in range(cs):
                for lz in range(cs):
                    wx, wz = x0 + lx, z0 + lz
                    w.set_block_raw(wx, 0, wz, solid_mat)
                    if lx == cs - 1 and lz == 2:
                        # Cross-chunk spillover write into adjacent chunk (cx+1, cz) before it generates
                        w.set_block_raw(wx + 1, 3, wz, solid_mat)
            w.set_block_raw(x0 + 2, 1, z0 + 2, water_mat)

        world = ChunkedVoxelWorld(chunk_size=8, generator_fn=simple_gen)
        created = world.update_chunks_around(0.0, 0.0, load_radius=1, unload_radius=2, max_new_per_call=16)
        self.assertGreaterEqual(created, 5)
        self.assertTrue(world.has_block(0, 0, 0))
        self.assertTrue(world.is_solid(0, 0, 0))
        # Adjacent chunk (1, 0) must still have generated full ground even after spillover write at (8, 3, 2)
        self.assertTrue(world.has_block(8, 0, 0))
        self.assertTrue(world.has_block(8, 3, 2))
        # Water block is non-solid
        self.assertTrue(world.has_block(2, 1, 2))
        self.assertFalse(world.is_solid(2, 1, 2))

        # User edit persists across chunk unload & reload!
        custom_mat = VoxelMaterial(top=(255, 210, 40), side=(255, 210, 40), bottom=(255, 210, 40), name="Gold")
        world.set_block(1, 2, 1, custom_mat)
        world.remove_block(1, 0, 1)
        # Move far away to unload chunk (0, 0)
        world.update_chunks_around(80.0, 80.0, load_radius=1, unload_radius=2, max_new_per_call=16)
        self.assertNotIn((0, 0), world.chunks)
        # Return to (0, 0) and confirm overrides survived
        world.update_chunks_around(0.0, 0.0, load_radius=1, unload_radius=2, max_new_per_call=16)
        self.assertIn((0, 0), world.chunks)
        self.assertEqual(world.get_block(1, 2, 1), custom_mat)
        self.assertFalse(world.has_block(1, 0, 1))

        # Raycast skips non-solid water by default and hits solid ground beneath
        hit_solid = world.raycast(origin=(2.5, 4.0, 2.5), direction=(0.0, -1.0, 0.0), max_distance=8.0, skip_non_solid=True)
        self.assertIsNotNone(hit_solid)
        assert hit_solid is not None
        self.assertEqual(hit_solid.block_pos, (2, 0, 2))

        # Raycast with skip_non_solid=False hits the water at y=1
        hit_water = world.raycast(origin=(2.5, 4.0, 2.5), direction=(0.0, -1.0, 0.0), max_distance=8.0, skip_non_solid=False)
        self.assertIsNotNone(hit_water)
        assert hit_water is not None
        self.assertEqual(hit_water.block_pos, (2, 1, 2))

        # Physics body passes through non-solid water and lands on solid ground y=0 (top at 1.0)
        body = VoxelPhysicsBody(position=(2.5, 2.8, 2.5), height=1.6)
        for _ in range(45):
            body.step(world, dt=1.0 / 60.0)
        self.assertTrue(body.on_ground)
        self.assertAlmostEqual(body.aabb.min_y, 1.0, delta=0.08)

    def test_11_skydome_day_night_cycle_and_sync(self) -> None:
        sky = SkyDome3D(time_of_day=0.25, day_speed=0.1)
        light = DirectionalLight()
        fog = Fog3D(enabled=True)
        cam = Camera3D(position=(0.0, 5.0, 0.0), yaw=0.0, pitch=0.2)

        pal_noon = sky.sync_lighting(light, fog)
        self.assertGreater(float(pal_noon["day_factor"]), 0.8)
        sky.draw(self.screen, cam)

        sky.time_of_day = 0.75  # midnight
        pal_night = sky.sync_lighting(light, fog)
        self.assertGreater(float(pal_night["night_factor"]), 0.8)
        sky.draw(self.screen, cam)

        sky.update(0.5)
        self.assertAlmostEqual(sky.time_of_day, 0.80, places=4)

    def test_12_textured_zbuffer_renderer_and_clipping(self) -> None:
        # Textured polygon near-plane clipping preserves valid UVs
        poly_c = np.array([[-1.0, -1.0, -0.4], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, -1.0, -0.4]], dtype=np.float32)
        cp, cuv = clip_textured_polygon_near_plane(poly_c, CANONICAL_QUAD_UV, near_z=0.2)
        self.assertGreaterEqual(len(cp), 3)
        self.assertEqual(len(cp), len(cuv))
        self.assertTrue(np.all(cp[:, 2] >= 0.1999))
        self.assertTrue(np.all((cuv >= -1e-4) & (cuv <= 1.0001)))

        # Z-buffer occlusion check: draw FAR red triangle AFTER NEAR blue triangle -> NEAR blue wins!
        rast = SoftwareTexturedRasterizer(64, 64)
        rast.begin_frame(clear_color=(0, 0, 0))
        rast.draw_triangle((10.0, 10.0, 0.5), (50.0, 10.0, 0.5), (30.0, 50.0, 0.5), solid_color=(0, 0, 240))
        rast.draw_triangle((10.0, 10.0, 0.2), (50.0, 10.0, 0.2), (30.0, 50.0, 0.2), solid_color=(240, 0, 0))
        center_px = tuple(int(c) for c in rast.color_buf[25, 30])
        self.assertEqual(center_px, (0, 0, 240))

        # Full textured Renderer3D with ChunkedVoxelWorld + SkyDome3D + transparent glass + Mesh3D
        tex_arr = np.zeros((16, 16, 3), dtype=np.uint8)
        tex_arr[:, :] = (90, 175, 70)
        tex_arr[::2, ::2] = (65, 140, 50)
        tex_grass = Texture2D(tex_arr)
        glass_rgba = np.full((16, 16, 4), (170, 225, 255, 95), dtype=np.uint8)
        glass_rgba[0, :, 3] = 255
        glass_rgba[-1, :, 3] = 255
        glass_rgba[:, 0, 3] = 255
        glass_rgba[:, -1, 3] = 255
        tex_glass = Texture2D(glass_rgba)

        grass_mat = VoxelMaterial(top_texture=tex_grass, side_texture=tex_grass, bottom_texture=tex_grass, name="Grass")
        glass_mat = VoxelMaterial(top_texture=tex_glass, side_texture=tex_glass, bottom_texture=tex_glass, alpha=200, name="Glass")

        world = ChunkedVoxelWorld(chunk_size=8)
        for x in range(4):
            for z in range(4):
                world.set_block(x, 0, z, grass_mat)
        world.set_block(2, 1, 2, glass_mat)

        scene = Scene3D()
        scene.add_mesh(Primitives.cube(size=0.8, color=(255, 199, 51), position=(1.0, 2.0, 2.0)))

        cam = Camera3D(position=(2.0, 3.2, -3.5), yaw=0.0, pitch=-0.35, fov=72.0)
        sky = SkyDome3D(time_of_day=0.25)
        renderer = Renderer3D(800, 600, textured=True, render_scale=0.45, texture_lod_distance=20.0)
        sky.sync_lighting(renderer.light, renderer.fog)

        drawn = renderer.render_scene(self.screen, cam, scene=scene, voxel_world=world, sky_dome=sky)
        self.assertGreater(drawn, 4)

    def test_13_native_and_numpy_backend_parity_and_frustum_culling(self) -> None:
        from pyforge3d import HAS_NATIVE_RASTERIZER, TextureAtlas

        tex_arr = np.zeros((16, 16, 3), dtype=np.uint8)
        tex_arr[:, :] = (82, 170, 62)
        tex_arr[0:4, :] = (110, 200, 75)
        tex = Texture2D(tex_arr)
        atlas = TextureAtlas()
        tid = atlas.register(tex)
        self.assertEqual(tid, 0)
        self.assertEqual(atlas.register(tex), 0)
        atlas.ensure_packed()
        self.assertEqual(len(atlas), 1)

        mat = VoxelMaterial(top_texture=tex, side_texture=tex, bottom_texture=tex, name="Block")
        world = ChunkedVoxelWorld(chunk_size=8)
        for x in range(-12, 16):
            for z in range(-12, 16):
                world.set_block_raw(x, 0, z, mat)
        world.get_culled_render_data()

        cam = Camera3D(position=(2.0, 4.0, 2.0), yaw=0.0, pitch=-0.25, fov=72.0, far=24.0)
        sky = SkyDome3D(time_of_day=0.25)

        surf_native = pygame.Surface((400, 240), depth=32)
        surf_numpy = pygame.Surface((400, 240), depth=32)

        r_native = Renderer3D(400, 240, textured=True, render_scale=0.5, use_native=True)
        r_numpy = Renderer3D(400, 240, textured=True, render_scale=0.5, use_native=False)
        sky.sync_lighting(r_native.light, r_native.fog)
        sky.sync_lighting(r_numpy.light, r_numpy.fog)

        n_nat = r_native.render_scene(surf_native, cam, voxel_world=world, sky_dome=sky)
        n_np = r_numpy.render_scene(surf_numpy, cam, voxel_world=world, sky_dome=sky)
        self.assertGreater(n_nat, 4)
        self.assertEqual(n_nat, n_np)
        self.assertTrue(HAS_NATIVE_RASTERIZER)

        # Verify pixel parity between native C99 and Pure-NumPy fallback!
        arr_nat = pygame.surfarray.array3d(surf_native).astype(np.int16)
        arr_np = pygame.surfarray.array3d(surf_numpy).astype(np.int16)
        mean_diff = float(np.abs(arr_nat - arr_np).mean())
        self.assertLess(mean_diff, 4.0)

    def test_14_textured_mesh_voxel_overlays_and_serialization(self) -> None:
        # 1. Textured Mesh3D (Primitives.textured_cube) & Voxel Block Overlays in BOTH C99 and NumPy backends
        top_tex = Texture2D(np.full((16, 16, 3), (80, 185, 70), dtype=np.uint8))
        side_tex = Texture2D(np.full((16, 16, 3), (140, 100, 65), dtype=np.uint8))
        crack_rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        crack_rgba[4:12, 7:9] = (15, 15, 15, 210)
        crack_rgba[7:9, 4:12] = (15, 15, 15, 210)
        crack_tex = Texture2D(crack_rgba)
        self.assertTrue(crack_tex.has_cutout)

        mini_cube = Primitives.textured_cube(size=0.35, top_texture=top_tex, side_texture=side_tex, position=(1.5, 1.8, 2.5))
        self.assertEqual(len(mini_cube.face_textures), 6)
        scene = Scene3D()
        scene.add_mesh(mini_cube)

        mat = VoxelMaterial(top_texture=top_tex, side_texture=side_tex, bottom_texture=side_tex, name="Block")
        world = ChunkedVoxelWorld(chunk_size=8)
        world.set_block(1, 0, 2, mat)
        world.set_block(2, 0, 2, mat)
        world.remove_block(2, 0, 2)

        cam = Camera3D(position=(0.2, 2.4, -1.2), fov=72.0)
        cam.look_at((1.5, 1.0, 2.5))
        overlays = {(1, 0, 2): crack_tex}
        surfs = []
        for use_native in (True, False):
            surf = pygame.Surface((320, 200), depth=32)
            rend = Renderer3D(320, 200, textured=True, render_scale=0.5, use_native=use_native)
            faces = rend.render_scene(surf, cam, scene=scene, voxel_world=world, block_overlays=overlays)
            self.assertGreaterEqual(faces, 7)
            surfs.append(pygame.surfarray.array3d(surf).astype(np.int16))
        self.assertLess(float(np.abs(surfs[0] - surfs[1]).mean()), 4.0)

        # 2. ChunkedVoxelWorld override export / import round-trip
        records = world.export_overrides(lambda m: m.name)
        self.assertEqual(len(records), 2)

        world2 = ChunkedVoxelWorld(chunk_size=8)
        restored = world2.import_overrides(records, lambda name: mat if name == "Block" else None)
        self.assertEqual(restored, 2)
        world2.generate_chunk(0, 0)
        self.assertEqual(world2.get_block(1, 0, 2), mat)
        self.assertFalse(world2.has_block(2, 0, 2))

    def test_15_greedy_meshing_polygon_reduction_and_tiled_parity(self) -> None:
        # Build a textured 8x8 flat floor + 4x3 wall in two worlds: greedy vs non-greedy!
        tex_arr = np.zeros((16, 16, 3), dtype=np.uint8)
        tex_arr[:, :] = (90, 170, 62)
        tex_arr[0:4, :] = (120, 200, 80)
        tex_arr[12:16, 12:16] = (40, 80, 35)
        tex = Texture2D(tex_arr)
        mat = VoxelMaterial(top_texture=tex, side_texture=tex, bottom_texture=tex, name="Tiled")

        w_greedy = ChunkedVoxelWorld(chunk_size=8, greedy_meshing=True)
        w_naive = ChunkedVoxelWorld(chunk_size=8, greedy_meshing=False)
        for x in range(8):
            for z in range(8):
                w_greedy.set_block_raw(x, 0, z, mat)
                w_naive.set_block_raw(x, 0, z, mat)
        for x in range(2, 6):
            for y in range(1, 4):
                w_greedy.set_block_raw(x, y, 5, mat)
                w_naive.set_block_raw(x, y, 5, mat)

        d_greedy = w_greedy.get_culled_render_data()
        d_naive = w_naive.get_culled_render_data()
        # Greedy meshing must slash face count by > 4x!
        self.assertLess(len(d_greedy["quads"]) * 4, len(d_naive["quads"]))

        # Visually compare rendered pixels between greedy-tiled and un-merged (must match closely!)
        cam = Camera3D(position=(4.0, 4.2, -3.5), yaw=0.0, pitch=-0.35, fov=72.0)
        for use_native in (True, False):
            s_g = pygame.Surface((320, 200), depth=32)
            s_n = pygame.Surface((320, 200), depth=32)
            r_g = Renderer3D(320, 200, textured=True, render_scale=0.5, use_native=use_native)
            r_n = Renderer3D(320, 200, textured=True, render_scale=0.5, use_native=use_native)
            r_g.fog.enabled = False
            r_n.fog.enabled = False
            r_g.render_scene(s_g, cam, voxel_world=w_greedy)
            r_n.render_scene(s_n, cam, voxel_world=w_naive)
            arr_g = pygame.surfarray.array3d(s_g).astype(np.int16)
            arr_n = pygame.surfarray.array3d(s_n).astype(np.int16)
            self.assertLess(float(np.abs(arr_g - arr_n).mean()), 4.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
