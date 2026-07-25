"""Post-processing for skeleton graphs.

Cut degree-3 branches and split degree-4 crossings based on the angles between
their incident edges. Short dead-end branches can also be pruned by length.
"""

from __future__ import annotations

import numpy as np

from ..graph import ContractionGraph, UndirectedGraph, connected_components
from ..utils import UnionFind
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


def _compact(vertices, edges, radii):
    used = np.zeros(len(vertices), dtype=bool)
    if len(edges):
        used[edges.reshape(-1)] = True
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    if radii is not None:
        radii = radii[used]
    return vertices[used], remap[edges], radii


def _split_degree3(v, graph, vertices, direction_span=1, min_branch_angle=30.0):
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


def _split_degree4(v, graph, vertices, direction_span=1, min_through_angle=160.0):
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


def clean_filament_graph(
    vertices: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray | None = None,
    *,
    direction_span: int = 5,
    min_through_angle: float = 160.0,
    min_branch_angle: float = 30.0,
    tick_length: float = 0.0,
    join_dist: float = 0.0,
    min_join_angle: float = 175.0,
    save_intermediates: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Clean a skeleton graph by cutting branches and splitting crossings.

    Postprocessing steps, in order:

    1. If ``tick_length > 0``, prune dead-end branches shorter than it via
       :func:`remove_ticks`.
    2. At each degree-3 junction, identify the straightest pair of arms. If the
       remaining arm diverges by at least ``min_branch_angle``, remove its edge
       adjacent to the junction. This cuts the arm from the through-going
       filament and removes its first segment. A one-edge arm is removed
       completely.
    3. Split each degree-4 crossing, separating its two through pairs when they
       are collinear to within ``min_through_angle``.
    4. If ``join_dist > 0``, join collinear endpoints across gaps up to
       this distance via :func:`join_close_components`.

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
        Minimum angle in degrees between a degree-3 node's odd arm and its
        through pair. When the angle meets this threshold, the edge adjacent
        to the junction on the odd arm is removed.
    tick_length:
        If > 0, prune dead-end branches shorter than this (physical) distance.
    join_dist:
        If > 0, join collinear endpoints across gaps up to this (physical)
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
    degrees = graph.node_degrees()

    splits, prune_edges = [], set()
    for v in np.where(degrees == 3)[0]:
        ids = _split_degree3(v, graph, vertices, direction_span, min_branch_angle)
        if ids:
            prune_edges.update(ids)
    for v in np.where(degrees == 4)[0]:
        ids = _split_degree4(v, graph, vertices, direction_span, min_through_angle)
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

    vertices, edges, radii = _compact(vertices, edges, radii)
    _snapshot("split")

    if join_dist > 0:
        vertices, edges, radii = join_close_components(
            vertices, edges, join_dist,
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


def _nodes_in_noncycle_components(indptr, dst, degrees):
    eligible = np.zeros(len(degrees), dtype=bool)
    visited = np.zeros(len(degrees), dtype=bool)
    for start in range(len(degrees)):
        if visited[start]:
            continue
        component = []
        stack = [start]
        visited[start] = True
        has_critical_node = False
        while stack:
            node = stack.pop()
            component.append(node)
            has_critical_node |= degrees[node] != 2
            for index in range(indptr[node], indptr[node + 1]):
                neighbor = int(dst[index])
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if has_critical_node:
            eligible[component] = True
    return eligible


def remove_ticks(vertices, edges, tick_length, radii=None):
    """Prune short dead-end branches ("ticks") from a skeleton graph.

    A distance graph is built over the critical
    points (terminals, degree 1; branch points, degree >= 3), whose superedges
    are the paths between them weighted by physical length. The shortest terminal
    branch below ``tick_length`` is pruned repeatedly; when a branch point drops
    to degree 2 its two superedges are fused into one, so a real filament end is
    re-measured rather than clipped. Standalone paths (both ends terminal) are
    never pruned.

    Parameters
    ----------
    vertices:
        Float array with shape ``(V, D)`` of skeleton vertex coordinates.
    edges:
        Integer array with shape ``(E, 2)`` indexing ``vertices``. Self-edges
        and duplicate undirected edges are not supported.
    tick_length:
        Maximum branch length (physical) that may be pruned.
    radii:
        Optional per-vertex radii, carried through the same remapping.

    Returns
    -------
    vertices, edges, radii:
        The pruned arrays, reindexed with unused vertices dropped. ``radii`` is
        ``None`` when no input radii were given.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2:
        raise ValueError(f"vertices must be a 2D array, got ndim={vertices.ndim}")
    raw_edges = np.asarray(edges)
    if not np.issubdtype(raw_edges.dtype, np.integer):
        raise TypeError(f"edges must have an integer dtype, got dtype={raw_edges.dtype}")
    if raw_edges.ndim != 2 or raw_edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got shape={raw_edges.shape}")
    if np.issubdtype(raw_edges.dtype, np.signedinteger) and np.any(raw_edges < 0):
        raise ValueError("edges must not contain negative node ids")
    edges_u64 = np.ascontiguousarray(raw_edges, dtype=np.uint64)
    num_nodes = len(vertices)
    if edges_u64.size and int(edges_u64.max()) >= num_nodes:
        raise IndexError(
            f"edge endpoint must be smaller than len(vertices)={num_nodes}"
        )
    if radii is not None and len(radii) != num_nodes:
        raise ValueError(
            f"radii length must be {num_nodes}, got length={len(radii)}"
        )
    edges = edges_u64.astype(np.int64, copy=False)
    if len(edges) == 0:
        return vertices, edges.copy(), radii

    graph = UndirectedGraph.from_edges(num_nodes, edges_u64)
    if graph.number_of_edges != len(edges):
        raise ValueError("edges must not contain duplicate undirected edges")

    indptr, dst, _, degrees = _adjacency(num_nodes, edges)
    lengths = np.linalg.norm(
        vertices[edges[:, 1]] - vertices[edges[:, 0]],
        axis=1,
    )
    work = ContractionGraph(graph, parallel_edges="keep")
    work.add_edge_values("length", lengths, reduction="sum")

    eligible = _nodes_in_noncycle_components(indptr, dst, degrees)
    for node in np.where((degrees == 2) & eligible)[0]:
        node = int(node)
        if work.can_suppress_node(node):
            work.suppress_node(node)

    threshold = float(tick_length)
    while True:
        edge_ids, active_lengths = work.active_edge_values("length")
        best = None
        for edge, length in zip(edge_ids, active_lengths, strict=True):
            edge = int(edge)
            length = float(length)
            a, b = work.uv(edge)
            terminal_a = work.degree(a) == 1
            terminal_b = work.degree(b) == 1
            if terminal_a ^ terminal_b and length < threshold:
                candidate = (length, edge, a, b)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None:
            break
        _, edge, a, b = best
        work.erase_edge(edge)
        for node in (a, b):
            if work.can_suppress_node(node):
                work.suppress_node(node)

    edges = edges[work.materialize().edge_mapping >= 0]
    vertices, edges, radii = _compact(vertices, edges, radii)
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


def join_close_components(vertices, edges, dist, *, min_join_angle=175.0,
                          direction_span=5, radii=None):
    """Reconnect fragmented filaments by joining collinear endpoints across gaps.

    Endpoints (degree = 1) of different connected components are joined with a
    new edge when they are within ``dist`` and the two fragments are nearly
    collinear through the gap: the outward tangent at each endpoint must point
    along the gap to within ``180 - min_join_angle`` degrees, so a straight
    continuation reads ~180 (``min_join_angle``). Joins are made shortest-first
    with a union-find so each pair of components is joined at most once and each
    endpoint is used once.

    Parameters
    ----------
    vertices, edges:
        Skeleton graph; ``edges`` indexes ``vertices``.
    dist:
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
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise ImportError(
            "join_close_components requires scipy; install scipy to use this function"
        ) from error

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
    for ia, ib in tree.query_pairs(dist):
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
    uf = UnionFind(int(labels.max()) + 1)

    used, new_edges = set(), []
    for _, a, b in candidates:
        if a in used or b in used:
            continue
        if uf.find(int(labels[a])) == uf.find(int(labels[b])):
            continue
        uf.merge(int(labels[a]), int(labels[b]))
        new_edges.append([a, b])
        used.add(a)
        used.add(b)

    if new_edges:
        edges = np.concatenate([edges, np.asarray(new_edges, dtype=np.int64)], axis=0)
    else:
        edges = edges.copy()
    return vertices, edges, radii


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
        :func:`bioimage_cpp.graph.connected_components`. Both endpoints of
        each edge should have the same label.
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
        delta = vi[b] - vi[a]
        steps = int(np.abs(delta).max()) + 1
        t = np.linspace(0.0, 1.0, steps)
        pts = np.rint(vi[a][None, :] + t[:, None] * delta[None, :]).astype(np.int64)
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


__all__ = ["clean_filament_graph", "draw_instances", "join_close_components", "remove_ticks"]
