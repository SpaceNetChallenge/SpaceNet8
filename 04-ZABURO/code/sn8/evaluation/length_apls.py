import copy
import heapq
import math
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


class Node:
    def __init__(self, coord: Tuple[float, float]) -> None:
        self.coord = coord
        self.edges: List[LineString] = []


def cut_linestring_by_distance(line: LineString, distance: float) -> Tuple[LineString, LineString]:
    assert distance <= line.length

    coords = [line.coords[0]]
    cum_d = 0.0
    for i, p2 in enumerate(line.coords[1:], 1):
        p1 = coords[-1]
        d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        if cum_d + d == distance:
            return LineString(coords + [line.coords[i]]), LineString(line.coords[i:])
        elif cum_d + d > distance:
            r = (distance - cum_d) / d
            assert 0 <= r <= 1

            pr = (p1[0] + r * (p2[0] - p1[0]), p1[1] + r * (p2[1] - p1[1]))
            return LineString(coords + [pr]), LineString([pr] + line.coords[i:])
        else:
            coords.append(p2)
            cum_d += d

    raise RuntimeError


class Graph:
    def __init__(
        self, roads: Sequence[LineString], pixel_size: float = 0.31, path_delta_m: float = 50, max_snap_distance_m=4
    ) -> None:
        self.nodes: List[Node] = []
        self.coord_to_node: Dict[Tuple[float, float], Node] = {}
        self.path_delta = path_delta_m / pixel_size
        self.max_snap_distance = max_snap_distance_m / pixel_size

        self._init_nodes(roads)
        self._simplify()
        self._smooth()

    def _add_node(self, coord: Tuple[float, float]) -> Node:
        assert coord not in self.coord_to_node
        node = Node(coord)
        self.nodes.append(node)
        self.coord_to_node[coord] = node
        return node

    def _remove_node(self, node: Node) -> None:
        self.nodes.remove(node)
        del self.coord_to_node[node.coord]

    def _add_edge(self, n1: Node, n2: Node, edge: LineString) -> None:
        if n1.coord == edge.coords[0]:
            assert n2.coord == edge.coords[-1]
        else:
            assert n1.coord == edge.coords[-1]
            assert n2.coord == edge.coords[0]

        n1.edges.append(edge)
        n2.edges.append(edge)

    def _remove_edge(self, n1: Node, n2: Node, edge: LineString) -> None:
        if n1.coord == edge.coords[0]:
            assert n2.coord == edge.coords[-1]
        else:
            assert n1.coord == edge.coords[-1]
            assert n2.coord == edge.coords[0]

        n1.edges.remove(edge)
        n2.edges.remove(edge)

    def _init_nodes(self, roads: Sequence[LineString]) -> None:
        for path in roads:
            prev_node = None
            for coord in path.coords:
                if coord in self.coord_to_node:
                    node = self.coord_to_node[coord]
                else:
                    node = self._add_node(coord)

                if prev_node is not None:
                    self._add_edge(prev_node, node, LineString([prev_node.coord, node.coord]))

                prev_node = node

    def _simplify(self) -> None:
        to_remove = []
        for node in self.nodes:
            if len(node.edges) != 2:
                continue

            e1, e2 = node.edges

            # keep self loop
            if e1.coords[0] == e1.coords[-1]:
                continue

            e1_coords = list(e1.coords)
            if e1_coords[0] == node.coord:
                e1_coords = e1_coords[::-1]

            e2_coords = list(e2.coords)
            if e2_coords[-1] == node.coord:
                e2_coords = e2_coords[::-1]

            assert e1_coords[-1] == node.coord
            assert e2_coords[0] == node.coord

            e_new = LineString(e1_coords + e2_coords[1:])

            n1 = self.coord_to_node[e_new.coords[0]]
            self._remove_edge(node, n1, e1)

            n2 = self.coord_to_node[e_new.coords[-1]]
            self._remove_edge(node, n2, e2)

            self._add_edge(n1, n2, e_new)

            to_remove.append(node)

        for node in to_remove:
            self._remove_node(node)

    def _smooth(self) -> None:
        all_edges = []
        for node in self.nodes:
            all_edges.extend(node.edges)

        processed = []
        for edge in all_edges:
            if edge in processed:
                continue

            length = edge.length

            if length < 0.75 * self.path_delta:
                continue

            n = 1 if length < self.path_delta else int(length / self.path_delta)
            dist = length / (n + 1)

            node1 = self.coord_to_node[edge.coords[0]]
            node2 = self.coord_to_node[edge.coords[-1]]
            self._remove_edge(node1, node2, edge)

            start = node1
            remaining_edge = edge
            for _ in range(n):
                e1, e2 = cut_linestring_by_distance(remaining_edge, dist)
                new_node = self._add_node(e2.coords[0])

                self._add_edge(start, new_node, e1)

                start = new_node
                remaining_edge = e2

            self._add_edge(start, node2, remaining_edge)

            processed.append(edge)

    def inject_node(self, coord: Tuple[float, float]) -> Optional[Node]:
        for node in self.nodes:
            if node.coord == coord:
                return node

        min_dist = self.max_snap_distance
        best_edge = None
        p = Point(coord)
        for node in self.nodes:
            for edge in node.edges:
                if edge.coords[0] != node.coord:
                    continue
                q = nearest_points(edge, p)[0]
                d = p.distance(q)

                if d < min_dist:
                    min_dist = d
                    best_edge = edge

        if best_edge is not None:
            assert best_edge.coords[0] != coord and best_edge.coords[-1] != coord

            q = nearest_points(best_edge, p)[0]
            if q.coords[0] in self.coord_to_node:
                return self.coord_to_node[q.coords[0]]

            e1, e2 = cut_linestring_by_distance(best_edge, best_edge.project(q))
            assert e1.coords[-1] == e2.coords[0]
            new_node = self._add_node(e1.coords[-1])

            n1 = self.coord_to_node[best_edge.coords[0]]
            n2 = self.coord_to_node[best_edge.coords[-1]]

            self._remove_edge(n1, n2, best_edge)
            self._add_edge(n1, new_node, e1)
            self._add_edge(new_node, n2, e2)

            return new_node
        else:
            return None

    def shortest_path_from(self, start: Tuple[float, float]) -> Mapping[Tuple[float, float], float]:
        distances = defaultdict(lambda: float("inf"))
        distances[start] = 0.0

        que = [(0, start)]
        heapq.heapify(que)

        while len(que) > 0:
            d, coord = heapq.heappop(que)
            if d > distances[coord]:
                continue
            for edge in self.coord_to_node[coord].edges:
                next_coord = edge.coords[0] if edge.coords[0] != coord else edge.coords[-1]
                next_d = d + edge.length
                if next_d < distances[next_coord]:
                    distances[next_coord] = next_d
                    heapq.heappush(que, (next_d, next_coord))
        return distances


def apls_score(g1: Graph, g2: Graph) -> float:
    if len(g1.nodes) == 0 and len(g2.nodes) == 0:
        return 1.0
    if len(g1.nodes) == 0 or len(g2.nodes) == 0:
        return 0.0

    g2 = copy.deepcopy(g2)

    node_map: Dict[Node, Node] = {}
    for p1 in g1.nodes:
        p2 = g2.inject_node(p1.coord)
        if p2 is not None:
            node_map[p1] = p2

    total_diff = 0.0
    route_count = 0
    for start1 in g1.nodes:
        distances_in_g1 = g1.shortest_path_from(start1.coord)

        if start1 not in node_map:
            for end1 in g1.nodes:
                if start1 is not end1 and distances_in_g1[end1.coord] < float("inf"):
                    total_diff += 1.0
                    route_count += 1.0
        else:
            start2 = node_map[start1]
            distances_in_g2 = g2.shortest_path_from(start2.coord)

            for end1 in g1.nodes:
                if start1 is end1:
                    continue

                d1 = distances_in_g1[end1.coord]
                if d1 < float("inf"):
                    route_count += 1.0

                    if end1 not in node_map:
                        total_diff += 1.0
                    else:
                        end2 = node_map[end1]

                        d2 = distances_in_g2[end2.coord]
                        if d1 == 0.0 and d2 == 0.0:
                            total_diff += 0.0
                        elif d1 == 0.0 or d2 == 0.0:
                            total_diff += 1.0
                        else:
                            total_diff += min(1, abs(d1 - d2) / d1)

    return 1 - total_diff / route_count


if __name__ == "__main__":
    g1 = Graph([LineString([(1, 1), (3, 1), (5, 1), (5, 3), (5, 4), (3, 4), (3, 1)])])
    g2 = Graph([LineString([(1, 1), (3, 1), (5, 1), (5, 3), (5, 4), (3, 4)]), LineString([(1, 2), (1, 3)])])

    assert apls_score(g1, g2) == 1.0
    assert apls_score(g2, g1) == 1.0 - ((9 - 5) / 9 * 2 + (1 - 0) / 1 * 2) / 4
