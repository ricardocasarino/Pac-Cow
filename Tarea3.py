# LIBRERÍAS
import pyglet
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.window import Window, key
from pyglet.gl import *
from pyglet.app import run
from pyglet import math as pyglet_math
from pyglet import clock
import numpy as np
import sys, os
import random

# MÓDULOS
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.helpers import init_pipeline, mesh_from_file
from utils.camera import FreeCamera
from utils.scene_graph import SceneGraph
from utils.drawables import Model, DirectionalLight, Material
from utils import shapes, colliders


# ── Constantes del mapa ───────────────────────────────────────────────
CELL        = 1.0
WALL_H      = 0.35
PLAYER_SPD  = 4.0
GHOST_SPD   = 2.5

CHAR_WALL   = '#'
CHAR_PELLET = '.'
CHAR_PLAYER = 'P'
CHAR_GHOST  = 'G'

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]


# ── Cámara cenital fija ───────────────────────────────────────────────
class MyCam(FreeCamera):
    def __init__(self, position=np.array([0,0,0]), camera_type="perspective"):
        super().__init__(position, camera_type)
        self.direction = np.array([0, 0, 0])
        self.speed     = 2

    def get_view(self):
        return super().get_view().flatten()

    def get_projection(self):
        return super().get_projection().flatten()

    def time_update(self, dt):
        self.update()
        dir = self.direction[0]*self.forward + self.direction[1]*self.right
        dir_norm = np.linalg.norm(dir)
        if dir_norm:
            dir /= dir_norm
        self.position += dir * self.speed * dt
        self.focus = self.position + self.forward


# ── Controla la ventana ───────────────────────────────────────────────
class Controller(Window):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time     = 0.0
        self.score    = 0
        self.lives    = 3
        self.next_dir = (0, 0)


WIDTH  = 900
HEIGHT = 700
window = Controller(WIDTH, HEIGHT, "Tarea 3 - Pac-Man")


# ── Lectura del mapa ──────────────────────────────────────────────────
def load_map(path):
    with open(path) as f:
        lines = f.read().splitlines()
    cols = max(len(l) for l in lines)
    grid = [l.ljust(cols) for l in lines]
    rows = len(grid)

    walls        = set()
    pellets      = set()
    player_start = (1, 1)
    ghost_starts = []

    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == CHAR_WALL:
                walls.add((r, c))
            elif ch == CHAR_PELLET:
                pellets.add((r, c))
            elif ch == CHAR_PLAYER:
                player_start = (r, c)
            elif ch == CHAR_GHOST:
                ghost_starts.append((r, c))

    return grid, pellets, walls, player_start, ghost_starts, rows, cols


def grid_to_world(row, col):
    return col * CELL, -row * CELL


def is_free(r, c, walls, rows, cols):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False
    return (r, c) not in walls


def wrap_cell(r, c, rows, cols):
    """Aplica el efecto túnel: si sale del mapa por un lado, aparece en el opuesto."""
    return r % rows, c % cols


# ── Batching con normales y UVs (requerido por shader Phong) ─────────
def build_batch_mesh(cells, sx, sy, sz, y_offset=0.0):

    base_pos = np.array(shapes.Cube["position"], dtype=np.float32).reshape(-1, 3)
    base_uv  = np.array(shapes.Cube["uv"],       dtype=np.float32).reshape(-1, 2)
    base_nor = np.array(shapes.Cube["normal"],    dtype=np.float32).reshape(-1, 3)
    base_idx = np.array(shapes.Cube["indices"],   dtype=np.uint32)

    all_pos, all_uv, all_nor, all_idx = [], [], [], []
    offset = 0

    for (r, c) in cells:
        wx, wz = grid_to_world(r, c)
        verts = base_pos.copy()
        verts[:, 0] = verts[:, 0] * sx + wx
        verts[:, 1] = verts[:, 1] * sy + y_offset
        verts[:, 2] = verts[:, 2] * sz + wz
        all_pos.append(verts.flatten())
        all_uv.append(base_uv.flatten())
        all_nor.append(base_nor.flatten())
        all_idx.append(base_idx + offset)
        offset += len(base_pos)

    if not all_pos:
        return None

    return Model(
        np.concatenate(all_pos).tolist(),
        uv_data    = np.concatenate(all_uv).tolist(),
        normal_data= np.concatenate(all_nor).tolist(),
        index_data = np.concatenate(all_idx).tolist()
    )


# ── Entidad móvil ─────────────────────────────────────────────────────
class Entity:
    def __init__(self, row, col, speed):
        self.row      = row
        self.col      = col
        self.speed    = speed
        x, z          = grid_to_world(row, col)
        self.vis_x    = float(x)
        self.vis_z    = float(z)
        self.cur_dir  = (0, 0)
        self.moving   = False
        self.progress = 0.0
        self.src_row  = row
        self.src_col  = col
        self.dst_row  = row
        self.dst_col  = col



if __name__ == "__main__":

    root_dir = os.path.dirname(__file__)

    # ── Pipeline Phong (igual que en los auxiliares) ──────────────────
    pipeline = init_pipeline(
        root_dir + "/shaders/phong.vert",
        root_dir + "/shaders/phong.frag"
    )


    # ── Cargar mapa ───────────────────────────────────────────────────
    grid, pellets, walls, player_start, ghost_starts, ROWS, COLS = \
        load_map(root_dir + "/level.txt")

    active_pellets = set(pellets)

    # ── Cámara centrada sobre el mapa ─────────────────────────────────
    cx = COLS * CELL / 2 - CELL / 2
    cz = -(ROWS * CELL / 2 - CELL / 2)

    cam = MyCam(np.array([cx, ROWS * 0.8, cz + ROWS * 0.2]))
    # Apuntar hacia el centro del mapa fijando yaw y pitch
    cam.pitch = -np.arctan2(ROWS * 0.8, ROWS * 0.2 + 0.001)
    cam.yaw   = -np.pi / 2
    cam.update()

    # ── SceneGraph ────────────────────────────────────────────────────
    world = SceneGraph(cam)

    # ── Luz direccional (igual que en los auxiliares) ─────────────────
    world.add_node("sun",
                   pipeline=pipeline,
                   light=DirectionalLight(
                       ambient =[0.4, 0.4, 0.4],
                       diffuse =[0.8, 0.8, 0.8],
                       specular=[0.3, 0.3, 0.3]
                   ),
                   rotation=[-np.pi/4, np.pi/4, 0])

    # ── Suelo ─────────────────────────────────────────────────────────
    floor_mesh = Model(
        shapes.Cube["position"],
        uv_data    =shapes.Cube["uv"],
        normal_data=shapes.Cube["normal"],
        index_data =shapes.Cube["indices"]
    )
    world.add_node("floor",
                   mesh=floor_mesh, pipeline=pipeline,
                   material=Material(ambient=[0.05,0.05,0.1], diffuse=[0.08,0.08,0.15]),
                   scale=[COLS*CELL, 0.05, ROWS*CELL],
                   position=[cx, -0.05, cz])

    # ── Paredes (batch) ───────────────────────────────────────────────
    wall_batch = build_batch_mesh(walls, CELL, WALL_H, CELL, y_offset=WALL_H/2)
    if wall_batch:
        world.add_node("walls_batch",
                       mesh=wall_batch, pipeline=pipeline,
                       material=Material(ambient=[0.05,0.05,0.5],
                                         diffuse=[0.1,0.1,0.9],
                                         specular=[0.3,0.3,0.3],
                                         shininess=32))

    # ── Pastillas (batch) ─────────────────────────────────────────────
    pellet_batch = build_batch_mesh(active_pellets, 0.18, 0.18, 0.18, y_offset=0.12)
    if pellet_batch:
        world.add_node("pellets_batch",
                       mesh=pellet_batch, pipeline=pipeline,
                       material=Material(ambient=[0.8,0.8,0.3],
                                         diffuse=[1.0,1.0,0.5],
                                         specular=[0.5,0.5,0.2],
                                         shininess=16))

    # ── Pac-Man (cow.obj) ────────────────────────────────────────────
    player_mesh = mesh_from_file(root_dir + "/cow.obj")[0]["mesh"]
    # Garantizar UVs para el shader Phong (que requiere texCoord)
    if player_mesh.uv_data is None:
        n_verts = len(player_mesh.position_data) // 3
        player_mesh.uv_data = [0.0, 0.0] * n_verts

    pr, pc = player_start
    player  = Entity(pr, pc, PLAYER_SPD)
    px, pz  = grid_to_world(pr, pc)
    world.add_node("player",
                   mesh=player_mesh, pipeline=pipeline,
                   material=Material(ambient=[0.8,0.8,0.0],
                                     diffuse=[1.0,1.0,0.0],
                                     specular=[0.5,0.5,0.0],
                                     shininess=32),
                   scale=[1, 1, 1],
                   position=[px, 0.3, pz])

    # ── Collider de Pac-Man ───────────────────────────────────────────
    half = 0.3
    player_collider = colliders.AABB("player", [-half,-half,-half], [half,half,half])
    player_collider.set_position([px, 0.3, pz])

    # ── Fantasmas (rat.obj) ────────────────────────────────────────
    ghost_base_mesh = mesh_from_file(root_dir + "/rat.obj")[0]["mesh"]
    # Garantizar UVs para el shader Phong (que requiere texCoord)
    if ghost_base_mesh.uv_data is None:
        n_verts = len(ghost_base_mesh.position_data) // 3
        ghost_base_mesh.uv_data = [0.0, 0.0] * n_verts

    ghost_materials = [
        Material(ambient=[0.8,0.1,0.1], diffuse=[1.0,0.2,0.2], specular=[0.3,0.1,0.1], shininess=16),
        Material(ambient=[0.8,0.4,0.6], diffuse=[1.0,0.6,0.9], specular=[0.3,0.2,0.3], shininess=16),
        Material(ambient=[0.0,0.5,0.8], diffuse=[0.0,0.8,1.0], specular=[0.1,0.3,0.4], shininess=16),
        Material(ambient=[0.8,0.4,0.0], diffuse=[1.0,0.5,0.0], specular=[0.3,0.2,0.0], shininess=16),
    ]

    ghosts = []
    collision_manager = colliders.CollisionManager()
    collision_manager.add_collider(player_collider)

    if not ghost_starts:
        ghost_starts = [(9,9),(9,10),(10,9),(10,10)]

    for i, (gr, gc) in enumerate(ghost_starts):
        if (gr, gc) in walls:
            for dr, dc in DIRS:
                if is_free(gr+dr, gc+dc, walls, ROWS, COLS):
                    gr, gc = gr+dr, gc+dc
                    break
        g    = Entity(gr, gc, GHOST_SPD)
        gx, gz = grid_to_world(gr, gc)
        name = f"ghost_{i}"
        world.add_node(name,
                       mesh=ghost_base_mesh, pipeline=pipeline,
                       material=ghost_materials[i % len(ghost_materials)],
                       scale=[1.4, 1.4, 1.4],
                       position=[gx, 0.3, gz])

        g_collider = colliders.AABB(name, [-half,-half,-half], [half,half,half])
        g_collider.set_position([gx, 0.3, gz])
        collision_manager.add_collider(g_collider)
        ghosts.append((name, g, g_collider))

    # ── Movimiento ────────────────────────────────────────────────────
    def move_entity(entity, dt, next_dir=None):
        if not entity.moving:
            for d in ([next_dir] if next_dir else []) + [entity.cur_dir]:
                if d == (0, 0):
                    continue
                nr = entity.row + d[0]
                nc = entity.col + d[1]
                if is_free(nr, nc, walls, ROWS, COLS):
                    entity.cur_dir  = d
                    entity.src_row  = entity.row
                    entity.src_col  = entity.col
                    entity.dst_row  = nr
                    entity.dst_col  = nc
                    entity.moving   = True
                    entity.progress = 0.0
                    break

        if entity.moving:
            entity.progress += entity.speed * dt
            sx, sz = grid_to_world(entity.src_row, entity.src_col)
            dx, dz = grid_to_world(entity.dst_row, entity.dst_col)
            if entity.progress >= 1.0:
                entity.row, entity.col = wrap_cell(entity.dst_row, entity.dst_col, ROWS, COLS)
                entity.moving   = False
                entity.progress = 0.0
                wx, wz = grid_to_world(entity.row, entity.col)
                entity.vis_x    = wx
                entity.vis_z    = wz
                return (entity.row, entity.col)
            else:
                t = entity.progress
                entity.vis_x = sx + (dx - sx) * t
                entity.vis_z = sz + (dz - sz) * t
        return None

    def ghost_ai(ghost):
        dr, dc   = ghost.cur_dir
        opposite = (-dr, -dc)
        options  = [d for d in DIRS
                    if d != opposite
                    and is_free(ghost.row+d[0], ghost.col+d[1], walls, ROWS, COLS)]
        if not options:
            options = [d for d in DIRS
                       if is_free(ghost.row+d[0], ghost.col+d[1], walls, ROWS, COLS)]
        if options:
            ghost.cur_dir = random.choice(options)

    def rebuild_pellet_batch():
        if "pellets_batch" in world:
            world.remove_node("pellets_batch")
        if active_pellets:
            new_batch = build_batch_mesh(active_pellets, 0.18, 0.18, 0.18, y_offset=0.12)
            if new_batch:
                world.add_node("pellets_batch",
                               mesh=new_batch, pipeline=pipeline,
                               material=Material(ambient=[0.8,0.8,0.3],
                                                 diffuse=[1.0,1.0,0.5],
                                                 specular=[0.5,0.5,0.2],
                                                 shininess=16))

    def check_ghost_collisions():
        """Detecta colisión Pac-Man con fantasmas usando CollisionManager."""
        hits = collision_manager.check_collision("player")
        if not hits:
            return False
        # Si alguna colisión es con un fantasma (no consigo mismo)
        for name in hits:
            if name.startswith("ghost_"):
                return True
        return False

    # ── Eventos ───────────────────────────────────────────────────────
    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.UP    or symbol == key.W:
            window.next_dir = ( 1,  0)
        if symbol == key.DOWN  or symbol == key.S:
            window.next_dir = (-1,  0)
        if symbol == key.LEFT  or symbol == key.A:
            window.next_dir = ( 0, -1)
        if symbol == key.RIGHT or symbol == key.D:
            window.next_dir = ( 0,  1)
        if symbol == key.ESCAPE:
            window.close()

    # ── Update ────────────────────────────────────────────────────────
    def update(dt):
        window.time += dt

        if window.lives <= 0:
            return

        # Mover Pac-Man
        arrived = move_entity(player, dt, next_dir=window.next_dir)
        if arrived:
            r, c = arrived
            if (r, c) in active_pellets:
                active_pellets.discard((r, c))
                window.score += 10
                window.set_caption(
                    f"Pac-Man  |  Score: {window.score}  |  Vidas: {window.lives}")
                rebuild_pellet_batch()

        world["player"]["position"] = [player.vis_x, 0.3, player.vis_z]
        player_collider.set_position([player.vis_x, 0.3, player.vis_z])

        # Mover fantasmas y actualizar sus colliders
        for name, ghost, g_col in ghosts:
            if not ghost.moving:
                ghost_ai(ghost)
            move_entity(ghost, dt)
            world[name]["position"] = [ghost.vis_x, 0.3, ghost.vis_z]
            g_col.set_position([ghost.vis_x, 0.3, ghost.vis_z])

        # Colisión Pac-Man vs fantasmas con CollisionManager
        if check_ghost_collisions():
            window.lives -= 1
            window.set_caption(
                f"Pac-Man  |  Score: {window.score}  |  Vidas: {window.lives}")
            player.row, player.col = player_start
            player.moving   = False
            player.progress = 0.0
            px2, pz2 = grid_to_world(*player_start)
            player.vis_x, player.vis_z = px2, pz2
            world["player"]["position"] = [px2, 0.3, pz2]
            player_collider.set_position([px2, 0.3, pz2])
            # Resetear fantasmas a sus posiciones iniciales
            for (name, ghost, g_col), (igr, igc) in zip(ghosts, ghost_starts):
                ghost.row, ghost.col = igr, igc
                ghost.moving   = False
                ghost.progress = 0.0
                ghost.cur_dir  = (0, 0)
                gx2, gz2 = grid_to_world(igr, igc)
                ghost.vis_x, ghost.vis_z = gx2, gz2
                world[name]["position"] = [gx2, 0.3, gz2]
                g_col.set_position([gx2, 0.3, gz2])

        if not active_pellets:
            window.set_caption(f"¡GANASTE!  Score: {window.score}")

        # La cámara no se mueve (fija cenital)
        world.update()

    # ── Draw ──────────────────────────────────────────────────────────
    @window.event
    def on_draw():
        window.clear()
        glClearColor(0.0, 0.0, 0.05, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        world.draw()

        glDisable(GL_DEPTH_TEST)

    clock.schedule_interval(update, 1/60)
    run()