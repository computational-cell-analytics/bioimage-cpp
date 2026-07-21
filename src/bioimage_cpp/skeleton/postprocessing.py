"""Post-processing for skeleton graphs.

Resolve junction nodes so each filament becomes its own connected component.
Degree-3 and degree-4 nodes are split or pruned based on the angles between their
incident edges: the straightest pair is kept as the through-going filament and
the remaining arm(s) are either separated, or for short dead-ends (spurs), removed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..graph import connected_components
from ._graph import skeleton_to_graph


def _tangent(v, first_neighbor, graph, vertices, span):
    prev, cur = v, int(first_neighbor)
    for _ in range(span - 1):
        nbrs = np.asarray(graph.node_adjacency(cur))[:, 0]
        if len(nbrs) != 2:
            break
        nxt = int(nbrs[0]) if int(nbrs[0]) != prev else int(nbrs[1])
        prev, cur = cur, nxt
    d = vertices[cur] - vertices[v]
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def _pair_angle(di, dj):
    return np.degrees(np.arccos(np.clip(di @ dj, -1.0, 1.0)))


def split_degree3(v, graph, vertices, direction_span=1, min_branch_angle=30.0):
    adj = np.asarray(graph.node_adjacency(int(v)))
    if adj.shape[0] != 3:
        return None
    neighbors, edge_ids = adj[:, 0], adj[:, 1]
    dirs = np.stack([_tangent(int(v), n, graph, vertices, direction_span) for n in neighbors])
    best, best_angle = None, -1.0
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        ang = _pair_angle(dirs[a], dirs[b])
        if ang > best_angle:
            best_angle, best = ang, (a, b)
    i, j = best
    odd = ({0, 1, 2} - {i, j}).pop()
    branch_angle = min(_pair_angle(dirs[odd], dirs[i]), _pair_angle(dirs[odd], dirs[j]))
    if branch_angle < min_branch_angle:
        return None
    return [int(edge_ids[odd])]


def split_degree4(v, graph, vertices, direction_span=1, min_through_angle=160.0):
    adj = np.asarray(graph.node_adjacency(int(v)))
    if adj.shape[0] != 4:
        return None
    neighbors, edge_ids = adj[:, 0], adj[:, 1]
    dirs = np.stack([_tangent(int(v), n, graph, vertices, direction_span) for n in neighbors])
    best, best_score, best_min = None, -np.inf, 0.0
    for (a, b), (c, d) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        ang1, ang2 = _pair_angle(dirs[a], dirs[b]), _pair_angle(dirs[c], dirs[d])
        if ang1 + ang2 > best_score:
            best_score, best, best_min = ang1 + ang2, ((a, b), (c, d)), min(ang1, ang2)
    if best_min < min_through_angle:
        return None
    (_, pair_b) = best
    return [int(edge_ids[k]) for k in pair_b]


def clean_graph(
    vertices: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray | None = None,
    *,
    direction_span: int = 5,
    min_through_angle: float = 160.0,
    min_branch_angle: float = 30.0,
    tick_length: float = 0.0,
    join_radius: float = 0.0,
    min_join_angle: float = 175.0,
    save_intermediates: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Split skeleton junctions so each filament is its own component.

    Postprocessing steps, in order:

    1. If ``tick_length > 0``, prune dead-end branches shorter than it via
       :func:`remove_ticks`.
    2. Split each degree-3 junction, separating the odd arm when it diverges
       from the through pair by at least ``min_branch_angle``.
    3. Split each degree-4 crossing, separating its two through pairs when they
       are collinear to within ``min_through_angle``.
    4. If ``join_radius > 0``, reconnect collinear endpoints across gaps less
       than this distance via :func:`join_close_components`.

    Parameters
    ----------
    vertices:
        Float array with shape ``(V, D)`` of skeleton vertex coordinates.
    edges:
        Integer array with shape ``(E, 2)`` indexing ``vertices``.
    radii:
        Optional per-vertex radii, carried through the same remapping.
    direction_span:
        Number of nodes over which each arm's tangent is measured.
    min_through_angle:
        Minimum through-pair angle (degrees) for a degree-4 crossing to split.
    min_branch_angle:
        Minimum angle (degrees) between a degree-3 node's odd arm and its
        through pair for the odd arm to be separated.
    tick_length:
        If > 0, prune dead-end branches shorter than this (physical) distance.
    join_radius:
        If > 0, reconnect collinear endpoints across gaps up to this (physical)
        distance.
    min_join_angle:
        Minimum straightness (degrees) required for a join; 180 is collinear.
    save_intermediates:
        If a list is given, ``(name, vertices, edges, radii)`` snapshots are
        appended after each step ("raw", "ticks", "split", "join").

    Returns
    -------
    vertices, edges, radii:
        The cleaned arrays, reindexed with unused vertices dropped. ``radii`` is
        ``None`` when no input radii were given.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).copy()

    def _snapshot(name):
        if save_intermediates is not None:
            save_intermediates.append((
                name, vertices.copy(), edges.copy(),
                None if radii is None else np.asarray(radii).copy(),
            ))

    _snapshot("raw")
    if tick_length > 0:
        vertices, edges, radii = remove_ticks(vertices, edges, tick_length, radii=radii)
        edges = edges.copy()
    _snapshot("ticks")
    graph = skeleton_to_graph(vertices, edges)
    degrees = np.bincount(edges.reshape(-1), minlength=len(vertices))

    splits, prune_edges = [], set()
    for v in np.where(degrees == 3)[0]:
        ids = split_degree3(v, graph, vertices, direction_span, min_branch_angle)
        if ids:
            prune_edges.update(ids)
    for v in np.where(degrees == 4)[0]:
        ids = split_degree4(v, graph, vertices, direction_span, min_through_angle)
        if ids:
            splits.append((int(v), ids))

    extra_vertices, extra_radii = [], []
    next_id = len(vertices)
    for node, edge_ids in splits:
        dup = next_id
        next_id += 1
        extra_vertices.append(vertices[node])
        if radii is not None:
            extra_radii.append(radii[node])
        for e in edge_ids:
            edges[e] = np.where(edges[e] == node, dup, edges[e])
    if extra_vertices:
        vertices = np.concatenate([vertices, np.asarray(extra_vertices)], axis=0)
        if radii is not None:
            radii = np.concatenate([radii, np.asarray(extra_radii)])

    if prune_edges:
        keep = np.ones(len(edges), dtype=bool)
        keep[list(prune_edges)] = False
        edges = edges[keep]

    used = np.zeros(len(vertices), dtype=bool)
    if len(edges):
        used[edges.reshape(-1)] = True
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    vertices = vertices[used]
    if radii is not None:
        radii = radii[used]
    edges = remap[edges]
    _snapshot("split")

    if join_radius > 0:
        vertices, edges, radii = join_close_components(
            vertices, edges, join_radius,
            min_join_angle=min_join_angle, direction_span=direction_span, radii=radii,
        )
    _snapshot("join")
    return vertices, edges, radii


def _adjacency(num_nodes, edges):
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    eid = np.concatenate([np.arange(len(edges)), np.arange(len(edges))])
    order = np.argsort(src, kind="stable")
    dst, eid = dst[order], eid[order]
    degrees = np.bincount(src, minlength=num_nodes)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    return indptr, dst, eid, degrees


def remove_ticks(vertices, edges, tick_length, radii=None):
    """Remove short dead-end branches ("ticks") from a skeleton graph.

    Ports kimimaro's `remove_ticks`. A distance graph is built over the critical
    points (terminals, degree 1; branch points, degree >= 3), whose superedges
    are the paths between them weighted by physical length. The shortest terminal
    branch below ``tick_length`` is removed repeatedly; when a branch point drops
    to degree 2 its two superedges are fused into one, so a real filament end is
    re-measured rather than clipped. Standalone paths (both ends terminal) are
    never removed.

    Parameters
    ----------
    vertices:
        Float array with shape ``(V, D)`` of skeleton vertex coordinates.
    edges:
        Integer array with shape ``(E, 2)`` indexing ``vertices``.
    tick_length:
        Maximum branch length (physical) that may be culled.
    radii:
        Optional per-vertex radii, carried through the same remapping.

    Returns
    -------
    vertices, edges, radii:
        The pruned arrays, reindexed with unused vertices dropped. ``radii`` is
        ``None`` when no input radii were given.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    num_nodes = len(vertices)
    if len(edges) == 0:
        return vertices, edges.copy(), radii

    indptr, dst, eid, degrees = _adjacency(num_nodes, edges)

    # Distance supergraph: sid -> [end_a, end_b, length, edge_ids].
    supers = {}
    incident = defaultdict(set)
    edge_used = np.zeros(len(edges), dtype=bool)
    sid = 0
    for node in map(int, np.where(degrees != 2)[0]):
        for k in range(indptr[node], indptr[node + 1]):
            if edge_used[eid[k]]:
                continue
            prev, cur = node, int(dst[k])
            path = [int(eid[k])]
            length = float(np.linalg.norm(vertices[cur] - vertices[node]))
            while degrees[cur] == 2:
                s, e = indptr[cur], indptr[cur + 1]
                nbrs, eids = dst[s:e], eid[s:e]
                pick = 0 if int(nbrs[0]) != prev else 1
                path.append(int(eids[pick]))
                nxt = int(nbrs[pick])
                length += float(np.linalg.norm(vertices[nxt] - vertices[cur]))
                prev, cur = cur, nxt
            for pe in path:
                edge_used[pe] = True
            supers[sid] = [node, cur, length, path]
            incident[node].add(sid)
            incident[cur].add(sid)
            sid += 1

    dropped = set()
    while True:
        best, best_len = None, tick_length
        for s, (a, b, length, _) in supers.items():
            terminal_a, terminal_b = len(incident[a]) == 1, len(incident[b]) == 1
            if (terminal_a ^ terminal_b) and length < best_len:
                best, best_len = s, length
        if best is None:
            break
        a, b, length, path = supers.pop(best)
        incident[a].discard(best)
        incident[b].discard(best)
        dropped.update(path)
        for node in (a, b):
            if len(incident[node]) == 2:
                s1, s2 = incident[node]
                a1, b1, l1, p1 = supers.pop(s1)
                a2, b2, l2, p2 = supers.pop(s2)
                far1 = b1 if a1 == node else a1
                far2 = b2 if a2 == node else a2
                for x in (far1, far2, node):
                    incident[x].discard(s1)
                    incident[x].discard(s2)
                supers[sid] = [far1, far2, l1 + l2, p1 + p2]
                incident[far1].add(sid)
                incident[far2].add(sid)
                sid += 1

    if dropped:
        keep = np.ones(len(edges), dtype=bool)
        keep[list(dropped)] = False
        edges = edges[keep]
    else:
        edges = edges.copy()

    used = np.zeros(num_nodes, dtype=bool)
    if len(edges):
        used[edges.reshape(-1)] = True
    remap = np.full(num_nodes, -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    vertices = vertices[used]
    if radii is not None:
        radii = radii[used]
    edges = remap[edges]
    return vertices, edges, radii


def _endpoint_tangent(endpoint, indptr, dst, degrees, vertices, span):
    if degrees[endpoint] == 0:
        return None
    prev, cur = endpoint, int(dst[indptr[endpoint]])
    for _ in range(span - 1):
        if degrees[cur] != 2:
            break
        s, e = indptr[cur], indptr[cur + 1]
        nbrs = dst[s:e]
        nxt = int(nbrs[0]) if int(nbrs[0]) != prev else int(nbrs[1])
        prev, cur = cur, nxt
    direction = vertices[endpoint] - vertices[cur]      # outward: interior -> tip
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else None


def join_close_components(vertices, edges, radius, *, min_join_angle=175.0,
                          direction_span=5, radii=None):
    """Reconnect fragmented filaments by joining collinear endpoints across gaps.

    Endpoints (degree = 1) of different connected components are joined with a
    new edge when they are within ``radius`` and the two fragments are nearly
    collinear through the gap: the outward tangent at each endpoint must point
    along the gap to within ``180 - min_join_angle`` degrees, so a straight
    continuation reads ~180 (``min_join_angle``). Joins are made shortest-first
    with a union-find so each pair of components is linked at most once and each
    endpoint is used once.

    Parameters
    ----------
    vertices, edges:
        Skeleton graph; ``edges`` indexes ``vertices``.
    radius:
        Maximum gap (physical) across which endpoints may be joined.
    min_join_angle:
        Minimum straightness (degrees) of the joined path; 180 is perfectly
        collinear.
    direction_span:
        Nodes over which each endpoint's tangent is measured.
    radii:
        Optional per-vertex radii, returned unchanged.

    Returns
    -------
    vertices, edges, radii:
        ``vertices`` and ``radii`` unchanged; ``edges`` has the join edges added.
    """
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    n = len(vertices)
    if len(edges) == 0:
        return vertices, edges.copy(), radii

    indptr, dst, _, degrees = _adjacency(n, edges)
    labels = connected_components(skeleton_to_graph(vertices, edges))
    endpoints = np.where(degrees <= 1)[0]
    if len(endpoints) < 2:
        return vertices, edges.copy(), radii

    tol = np.deg2rad(180.0 - min_join_angle)
    tree = cKDTree(vertices[endpoints])
    candidates = []
    for ia, ib in tree.query_pairs(radius):
        a, b = int(endpoints[ia]), int(endpoints[ib])
        if labels[a] == labels[b]:
            continue
        gap = vertices[b] - vertices[a]
        dist = float(np.linalg.norm(gap))
        if dist == 0.0:
            continue
        ta = _endpoint_tangent(a, indptr, dst, degrees, vertices, direction_span)
        tb = _endpoint_tangent(b, indptr, dst, degrees, vertices, direction_span)
        if ta is None or tb is None:
            continue
        gdir = gap / dist
        bend_a = np.arccos(np.clip(ta @ gdir, -1.0, 1.0))
        bend_b = np.arccos(np.clip(tb @ -gdir, -1.0, 1.0))
        if bend_a <= tol and bend_b <= tol:
            candidates.append((dist, a, b))

    candidates.sort()
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    used, new_edges = set(), []
    for _, a, b in candidates:
        if a in used or b in used:
            continue
        ra, rb = find(int(labels[a])), find(int(labels[b]))
        if ra == rb:
            continue
        parent[ra] = rb
        new_edges.append([a, b])
        used.add(a)
        used.add(b)

    if new_edges:
        edges = np.concatenate([edges, np.asarray(new_edges, dtype=np.int64)], axis=0)
    else:
        edges = edges.copy()
    return vertices, edges, radii


def _line_voxels(start, stop):
    delta = stop - start
    steps = int(np.abs(delta).max()) + 1
    t = np.linspace(0.0, 1.0, steps)
    return np.rint(start[None, :] + t[:, None] * delta[None, :]).astype(np.int64)


def draw_instances(vertices, edges, labels, shape, radius=1):
    """Rasterize a labeled skeleton graph into a dense instance volume.

    Each edge is drawn as a line between its two vertices and dilated by a ball
    of the given radius. Every voxel on a component's tubes is set to that
    component's label plus one, so background stays zero.

    Parameters
    ----------
    vertices:
        Float array with shape ``(V, 3)`` of vertex coordinates in voxel index
        space matching ``shape`` (``(z, y, x)`` order).
    edges:
        Integer array with shape ``(E, 2)`` indexing ``vertices``.
    labels:
        Per-vertex integer labels, e.g. from
        :func:`bioimage_cpp.graph.connected_components`.
    shape:
        Output volume shape ``(Z, Y, X)``.
    radius:
        Tube radius in voxels.

    Returns
    -------
    numpy.ndarray
        Integer volume of ``shape`` with background ``0`` and each tube voxel
        set to ``labels[endpoint] + 1``.
    """
    vertices = np.asarray(vertices)
    edges = np.asarray(edges)
    labels = np.asarray(labels)
    shape = tuple(int(s) for s in shape)

    if len(edges) == 0:
        return np.zeros(shape, dtype=np.uint16)

    r = int(radius)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    offsets = np.stack(np.where(zz ** 2 + yy ** 2 + xx ** 2 <= r * r), axis=1) - r

    vi = np.rint(vertices).astype(np.int64)
    centers, center_labels = [], []
    for a, b in edges:
        pts = _line_voxels(vi[a], vi[b])
        centers.append(pts)
        center_labels.append(np.full(len(pts), int(labels[a]) + 1))
    centers = np.concatenate(centers)
    center_labels = np.concatenate(center_labels)

    dtype = np.uint16 if int(center_labels.max()) < 2 ** 16 else np.uint32
    volume = np.zeros(shape, dtype=dtype)
    shape_arr = np.asarray(shape)
    for offset in offsets:
        coords = centers + offset
        inside = ((coords >= 0) & (coords < shape_arr)).all(axis=1)
        c = coords[inside]
        volume[c[:, 0], c[:, 1], c[:, 2]] = center_labels[inside]
    return volume


__all__ = ["clean_graph", "draw_instances", "join_close_components", "remove_ticks"]
