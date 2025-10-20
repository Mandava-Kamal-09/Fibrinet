
from utils.logger.logger import Logger
from PIL import Image, ImageDraw, ImageFont
from src.managers.input.input_manager import InputManager
from src.managers.network.networks.base_network import BaseNetwork
from src.managers.network.degradation_engine.two_dimensional_spring_force_degradation_engine_without_biomechanics import TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics

import os
import csv


class CollapseAnalysisResult:
    def __init__(self, minimum_degradations:int, removal_order:list[int], collapse_metrics:dict):
        self.minimum_degradations = minimum_degradations
        self.removal_order = removal_order
        self.collapse_metrics = collapse_metrics


class CollapseAnalysisManager:
    """Compute minimal edge removals to collapse the network."""

    def __init__(self):
        self.input_manager = InputManager()
        self.physics_engine = TwoDimensionalSpringForceDegradationEngineWithoutBiomechanics()

    # PUBLIC API
    def find_minimum_degradations_to_collapse(self, excel_path:str, fast:bool=False, *, constrain_to_center_flush_region:bool=False, output_image_path:str|None=None) -> CollapseAnalysisResult:
        Logger.log(f"start find_minimum_degradations_to_collapse(self, {excel_path})")

        network = self.input_manager.get_network(excel_path)

        if constrain_to_center_flush_region:
            full_edge_map, full_adj = self._build_graph_maps(network)
            edge_id_to_uv, adj, flush_bounds = self._build_constrained_graph_maps(network)
        else:
            edge_id_to_uv, adj = self._build_graph_maps(network)
            flush_bounds = None

        if constrain_to_center_flush_region:
            min_k, optimal_order = self._constrained_min_cut(network, flush_bounds)
        else:
            greedy_order = self._greedy_bridge_first_order(adj, edge_id_to_uv)

            if fast:
                work = self._copy_adj(adj)
                taken = []
                for eid in greedy_order:
                    if self._is_collapsed(work):
                        break
                    if not self._edge_present(work, eid):
                        continue
                    self._remove_edge_in_adj(work, eid, edge_id_to_uv)
                    taken.append(eid)
                min_k = len(taken)
                optimal_order = taken
            else:
                min_k, optimal_order = self._guided_min_search(adj, edge_id_to_uv, greedy_order)

        spring_constant = network.meta_data.get("spring_stiffness_constant", "N/A")
        metrics = {
            "total_edges": len(edge_id_to_uv),
            "greedy_k": len(edge_id_to_uv) if constrain_to_center_flush_region else len(self._greedy_bridge_first_order(adj, edge_id_to_uv)),
            "mode": ("constrained-mincut" if constrain_to_center_flush_region else ("fast" if fast else "exact")),
            "spring_stiffness_constant": spring_constant,
        }
        if flush_bounds:
            metrics["flush_region_bounds"] = flush_bounds
        
        if output_image_path and flush_bounds is not None:
            try:
                cut_set = set(optimal_order) if constrain_to_center_flush_region else None
                self._render_flush_region_plot(
                    network,
                    flush_bounds,
                    set(edge_id_to_uv.keys()),
                    output_image_path,
                    cut_edge_ids=cut_set,
                )
                metrics["image_path"] = output_image_path
            except Exception as ex:
                Logger.log(f"Flush region image render failed: {ex}", Logger.LogPriority.ERROR)

        Logger.log(f"end find_minimum_degradations_to_collapse(self, excel_path)")
        return CollapseAnalysisResult(min_k, optimal_order, metrics)

    def iterate_constrained_cuts(self, excel_path: str, output_dir: str, max_steps: int = None) -> dict:
        """
        Iteratively remove edges from flush region using min-cut, one edge per iteration.
        
        Flow:
        1. Load input file
        2. Calculate flush region from initial network (constant)
        3. Generate initial flush region image
        4. Iterative cuts using fixed region edges
        5. Relax after each removal
        6. Output step images + CSV
        
        Args:
            excel_path: Path to network file
            output_dir: Directory for step images and CSV log
            max_steps: Optional override for max iterations (default: region edge count)
            
        Returns:
            dict with summary metrics and final state
        """
        Logger.log(f"start iterate_constrained_cuts(excel_path={excel_path}, output_dir={output_dir})")
        
        # Step 1: Load input file
        network = self.input_manager.get_network(excel_path)
        nodes = network.get_nodes()
        if not nodes:
            return {"error": "No nodes in network"}
        
        # Step 2: Calculate flush region from initial network (constant)
        x_vals = [n.n_x for n in nodes]
        y_vals = [n.n_y for n in nodes]
        xmin, xmax = min(x_vals), max(x_vals)
        ymin, ymax = min(y_vals), max(y_vals)
        
        # Calculate flush region bounds: centered with width = (xmax - xmin) / 2
        total_width = xmax - xmin
        flush_width = total_width / 2.0
        center_x = (xmin + xmax) / 2.0
        flush_xmin = center_x - flush_width / 2.0
        flush_xmax = center_x + flush_width / 2.0
        
        # Calculate fixed region edges from initial network
        initial_region_edges = self._get_region_edges(network, flush_xmin, flush_xmax, ymin, ymax)
        Logger.log(f"Network bounds: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")
        Logger.log(f"Flush region: x=[{flush_xmin:.2f}, {flush_xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")
        Logger.log(f"Initial region edges: {len(initial_region_edges)} edges")
        
        if not initial_region_edges:
            return {"error": "No edges in initial flush region"}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 3: Generate initial flush region image
        initial_image_path = os.path.join(output_dir, "initial_flush_region.png")
        try:
            self._render_initial_flush_region(network, flush_xmin, flush_xmax, ymin, ymax, 
                                            initial_region_edges, initial_image_path)
            Logger.log(f"Initial flush region image: {initial_image_path}")
        except Exception as ex:
            Logger.log(f"Initial image render failed: {ex}", Logger.LogPriority.ERROR)
            initial_image_path = "failed"
        
        # Initialize CSV log
        csv_path = os.path.join(output_dir, "iteration_log.csv")
        csv_headers = ["step", "removed_edge_id", "cut_size", "cumulative_removed", 
                      "spring_stiffness_constant", "x_min", "x_max", "y_min", "y_max", 
                      "lcc_nodes", "lcc_edges", "image_path"]
        
        spring_constant = network.meta_data.get("spring_stiffness_constant", "N/A")
        cumulative_removed = []
        step = 0
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
            
            # Step 4: Iterative cuts; region edges recomputed each iteration
            while True:
                step += 1
                Logger.log(f"Iteration {step}")
                
                # Set max_steps if not provided
                if max_steps is None:
                    max_steps = len(initial_region_edges)
                    
                if step > max_steps:
                    Logger.log(f"Reached max steps {max_steps} - stopping")
                    break
                
                # Recompute current region edges based on latest relaxed positions
                current_region_edges = self._get_region_edges(network, flush_xmin, flush_xmax, ymin, ymax)
                if not current_region_edges:
                    Logger.log("Region empty - stopping")
                    break
                
                # Compute min-cut using only current region edges
                cut_edges = self._compute_min_cut_from_region_edges(
                    network, flush_xmin, flush_xmax, current_region_edges
                )
                
                if not cut_edges:
                    # No s-t path across region; drain by removing lowest-ID region edge
                    Logger.log("No cut found - draining lowest-ID region edge")
                    cut_edges = [min(current_region_edges)]
                
                # Select edge with lowest ID from cut
                edge_to_remove = min(cut_edges)
                Logger.log(f"Removing edge {edge_to_remove}")
                
                # Remove edge from network
                self._remove_edge_from_network(network, edge_to_remove)
                cumulative_removed.append(edge_to_remove)
                
                # Step 5: Relax physics after each removal
                self.physics_engine.relax_network(network)
                
                # Compute LCC stats
                lcc_nodes, lcc_edges = self._get_lcc_stats(network)
                
                # Generate step image
                image_path = os.path.join(output_dir, f"step_{step:03d}.png")
                try:
                    self._render_iteration_plot(network, flush_xmin, flush_xmax, ymin, ymax, 
                                              current_region_edges, {edge_to_remove}, image_path)
                except Exception as ex:
                    Logger.log(f"Image render failed: {ex}", Logger.LogPriority.ERROR)
                    image_path = "failed"
                
                # Write CSV row
                writer.writerow([
                    step, edge_to_remove, len(cut_edges), len(cumulative_removed),
                    spring_constant, flush_xmin, flush_xmax, ymin, ymax,
                    lcc_nodes, lcc_edges, image_path
                ])
                csvfile.flush()
        
        Logger.log(f"end iterate_constrained_cuts - completed {step} iterations")
        return {
            "total_steps": step,
            "cumulative_removed": cumulative_removed,
            "csv_path": csv_path,
            "initial_image": initial_image_path,
            "final_lcc_nodes": lcc_nodes,
            "final_lcc_edges": lcc_edges
        }

    # INTERNALS
    def _build_graph_maps(self, network: BaseNetwork):
        edge_id_to_uv = {}
        adj = {}
        for node in network.get_nodes():
            adj[node.get_id()] = set()
        for edge in network.get_edges():
            u = int(edge.n_from)
            v = int(edge.n_to)
            eid = int(edge.get_id())
            edge_id_to_uv[eid] = (u, v)
            if u not in adj:
                adj[u] = set()
            if v not in adj:
                adj[v] = set()
            adj[u].add((v, eid))
            adj[v].add((u, eid))
        return edge_id_to_uv, adj

    def _build_constrained_graph_maps(self, network: BaseNetwork):
        """Adjacency using only edges intersecting a centered x-slab flush region."""
        nodes = network.get_nodes()
        if not nodes:
            return {}, {}

        x_vals = [n.n_x for n in nodes]
        y_vals = [n.n_y for n in nodes]
        xmin, xmax = min(x_vals), max(x_vals)
        ymin, ymax = min(y_vals), max(y_vals)

        total_width = xmax - xmin
        flush_width = total_width / 2.0
        center_x = (xmin + xmax) / 2.0
        flush_xmin = center_x - flush_width / 2.0
        flush_xmax = center_x + flush_width / 2.0

        def edge_intersects_flush(edge):
            a = network.get_node_by_id(edge.n_from)
            b = network.get_node_by_id(edge.n_to)
            if a is None or b is None:
                return False
            exmin = min(a.n_x, b.n_x)
            exmax = max(a.n_x, b.n_x)
            return not (exmax < flush_xmin or exmin > flush_xmax)

        edge_id_to_uv = {}
        adj = {node.get_id(): set() for node in nodes}

        for edge in network.get_edges():
            if not edge_intersects_flush(edge):
                continue
            u = int(edge.n_from)
            v = int(edge.n_to)
            eid = int(edge.get_id())
            edge_id_to_uv[eid] = (u, v)
            if u not in adj:
                adj[u] = set()
            if v not in adj:
                adj[v] = set()
            adj[u].add((v, eid))
            adj[v].add((u, eid))

        flush_bounds = {
            "x_min": flush_xmin,
            "x_max": flush_xmax,
            "y_min": ymin,
            "y_max": ymax,
        }

        return edge_id_to_uv, adj, flush_bounds

    # ------------------ Constrained Min-Cut (unit capacities) ------------------
    def _constrained_min_cut(self, network: BaseNetwork, flush_bounds: dict):
        """Compute min s-t cut constrained to region edges (unit capacities)."""
        nodes = network.get_nodes()
        if not nodes:
            return 0, []

        fxmin = flush_bounds["x_min"]
        fxmax = flush_bounds["x_max"]
        left_nodes = {n.get_id() for n in nodes if n.n_x <= fxmin}
        right_nodes = {n.get_id() for n in nodes if n.n_x >= fxmax}
        if not left_nodes or not right_nodes:
            return 0, []

        # Helper to test if edge is inside/intersects the flush region along x
        def edge_in_region(edge):
            a = network.get_node_by_id(edge.n_from)
            b = network.get_node_by_id(edge.n_to)
            if a is None or b is None:
                return False
            exmin = min(a.n_x, b.n_x)
            exmax = max(a.n_x, b.n_x)
            return not (exmax < fxmin or exmin > fxmax)

        # Build directed graph for max-flow (Dinic) with capacities
        from collections import defaultdict, deque

        graph = defaultdict(list)

        def add_edge(u, v, cap, eid=None):
            graph[u].append([v, cap, len(graph[v]), eid])
            graph[v].append([u, 0, len(graph[u]) - 1, None])

        # Super-source S connects to all left_nodes; super-sink T from all right_nodes
        S = 10**9 + 7
        T = 10**9 + 9

        BIG = 10**9  # effectively infinite capacity for non-removable edges

        for e in network.get_edges():
            u = int(e.n_from)
            v = int(e.n_to)
            if edge_in_region(e):
                cap = 1  # removable inside region
            else:
                cap = BIG  # non-removable outside region
            # Undirected -> two directed edges
            add_edge(u, v, cap, eid=int(e.get_id()))
            add_edge(v, u, cap, eid=int(e.get_id()))

        for u in left_nodes:
            add_edge(S, int(u), BIG)
        for v in right_nodes:
            add_edge(int(v), T, BIG)

        # Dinic max-flow
        def bfs():
            level = {S: 0}
            q = deque([S])
            while q:
                x = q.popleft()
                for y, cap, rev, _ in graph[x]:
                    if cap > 0 and y not in level:
                        level[y] = level[x] + 1
                        q.append(y)
            return level

        def dfs(x, T, f, ptr, level):
            if x == T:
                return f
            adj = graph[x]
            i = ptr.get(x, 0)
            while i < len(adj):
                y, cap, rev, _ = adj[i]
                if cap > 0 and level.get(y, -1) == level.get(x, 0) + 1:
                    pushed = dfs(y, T, min(f, cap), ptr, level)
                    if pushed:
                        # adjust
                        adj[i][1] -= pushed
                        graph[y][rev][1] += pushed
                        ptr[x] = i
                        return pushed
                i += 1
            ptr[x] = i
            return 0

        flow = 0
        while True:
            level = bfs()
            if T not in level:
                break
            ptr = {}
            while True:
                pushed = dfs(S, T, BIG, ptr, level)
                if not pushed:
                    break
                flow += pushed

        # After max-flow, compute min-cut set by reachability from S in residual
        visited = set()
        stack = [S]
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            for y, cap, _, _ in graph[x]:
                if cap > 0 and y not in visited:
                    stack.append(y)

        # Cut edges are residual forward edges that cross S->T frontier with unit caps
        cut_edges = set()
        for u, adjlist in graph.items():
            if u not in visited:
                continue
            for idx, (v, cap, rev, eid) in enumerate(adjlist):
                # find residual capacity edge; need original capacity
                # original capacity = cap + reverse.cap now
                rev_edge = graph[v][rev]
                orig_cap = cap + rev_edge[1]
                if v not in visited and orig_cap > 0 and eid is not None and orig_cap < BIG:
                    cut_edges.add(eid)

        return len(cut_edges), list(cut_edges)

    def _render_flush_region_plot(self, network: BaseNetwork, flush_bounds: dict, edge_ids_in_region: set[int], out_path: str, *, cut_edge_ids: set[int]|None=None):
        """
        Render the network and highlight the flush region to a PNG image.
        - All edges: light gray
        - Edges intersecting region: blue
        - Flush region: semi-transparent red rectangle
        - Nodes: black dots
        """
        nodes = network.get_nodes()
        if not nodes:
            raise ValueError("No nodes to render")

        # Compute bounds
        x_vals = [n.n_x for n in nodes]
        y_vals = [n.n_y for n in nodes]
        xmin, xmax = min(x_vals), max(x_vals)
        ymin, ymax = min(y_vals), max(y_vals)

        # Image settings
        img_w, img_h = 1200, 800
        margin = 50

        # Avoid zero-size
        span_x = max(1e-6, xmax - xmin)
        span_y = max(1e-6, ymax - ymin)

        def world_to_image(x: float, y: float) -> tuple[int,int]:
            # Normalize to [0,1]
            nx = (x - xmin) / span_x
            ny = (y - ymin) / span_y
            # Convert to pixels, flip y for image coordinates
            px = int(margin + nx * (img_w - 2 * margin))
            py = int(img_h - (margin + ny * (img_h - 2 * margin)))
            return px, py

        # Create image
        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Draw flush region (semi-transparent)
        fxmin, fxmax = flush_bounds["x_min"], flush_bounds["x_max"]
        fymin, fymax = flush_bounds["y_min"], flush_bounds["y_max"]
        p1 = world_to_image(fxmin, fymin)
        p2 = world_to_image(fxmax, fymax)
        # Ensure rectangle coords ordered (left, top, right, bottom)
        left = min(p1[0], p2[0]); right = max(p1[0], p2[0])
        top = min(p1[1], p2[1]); bottom = max(p1[1], p2[1])
        draw.rectangle([left, top, right, bottom], fill=(255, 0, 0, 40), outline=(255, 0, 0, 180), width=2)

        # Draw all edges (light gray), highlight constrained edges (blue)
        constrained_edge_ids = set(edge_ids_in_region)
        cut_ids = set(cut_edge_ids) if cut_edge_ids else set()
        for edge in network.get_edges():
            a = network.get_node_by_id(edge.n_from)
            b = network.get_node_by_id(edge.n_to)
            if a is None or b is None:
                continue
            pa = world_to_image(a.n_x, a.n_y)
            pb = world_to_image(b.n_x, b.n_y)
            eid = int(edge.get_id())
            if eid in cut_ids:
                color = (220, 30, 30)
                width = 4
            elif eid in constrained_edge_ids:
                color = (40, 90, 200)
                width = 2
            else:
                color = (60, 60, 60)
                width = 2
            draw.line([pa, pb], fill=color, width=width)
            # Label cut edges at their midpoint
            if eid in cut_ids and font is not None:
                mx = (pa[0] + pb[0]) // 2
                my = (pa[1] + pb[1]) // 2
                draw.text((mx + 4, my - 8), str(eid), fill=(220, 30, 30), font=font)

        # Draw nodes
        r = 3
        for n in nodes:
            px, py = world_to_image(n.n_x, n.n_y)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 0, 0))

        img.save(out_path)

    def _copy_adj(self, adj):
        return {u: set(neigh) for u, neigh in adj.items()}

    def _remove_edge_in_adj(self, adj, eid, edge_id_to_uv):
        if eid not in edge_id_to_uv:
            return
        u, v = edge_id_to_uv[eid]
        # remove from u
        adj[u] = {pair for pair in adj.get(u, set()) if pair[1] != eid}
        # remove from v
        adj[v] = {pair for pair in adj.get(v, set()) if pair[1] != eid}

    def _largest_component_stats(self, adj):
        visited = set()
        max_nodes = 0
        max_edges = 0
        for start in adj.keys():
            if start in visited:
                continue
            stack = [start]
            comp_nodes = set()
            comp_edges = set()
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp_nodes.add(u)
                for v, eid in adj.get(u, set()):
                    comp_edges.add(eid)
                    if v not in visited:
                        stack.append(v)
            # undirected edges counted twice in adjacency accumulation
            edge_count = len(comp_edges)
            if len(comp_nodes) > max_nodes or (len(comp_nodes) == max_nodes and edge_count > max_edges):
                max_nodes = len(comp_nodes)
                max_edges = edge_count
        return max_nodes, max_edges

    def _is_collapsed(self, adj):
        # After physics relaxation, topology remains; criteria based on structure
        _, max_edges = self._largest_component_stats(adj)
        return max_edges <= 1

    def _bridges(self, adj):
        # Tarjan's algorithm for bridges
        time = 0
        disc = {}
        low = {}
        parent = {}
        bridges = []

        def dfs(u):
            nonlocal time
            time += 1
            disc[u] = low[u] = time
            for v, eid in adj.get(u, set()):
                if v not in disc:
                    parent[v] = (u, eid)
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.append(eid)
                elif parent.get(u, (None, None))[0] != v:
                    low[u] = min(low[u], disc[v])

        for u in adj.keys():
            if u not in disc:
                dfs(u)
        return bridges

    def _greedy_bridge_first_order(self, adj, edge_id_to_uv):
        order = []
        work = self._copy_adj(adj)
        for _ in range(len(edge_id_to_uv)):
            if self._is_collapsed(work):
                break
            bridges = self._bridges(work)
            candidate = None
            if bridges:
                # choose bridge that most reduces LCC size
                best_score = -1
                for eid in bridges:
                    tmp = self._copy_adj(work)
                    self._remove_edge_in_adj(tmp, eid, edge_id_to_uv)
                    nodes_before, edges_before = self._largest_component_stats(work)
                    nodes_after, edges_after = self._largest_component_stats(tmp)
                    score = (nodes_before - nodes_after) * 1000 + (edges_before - edges_after)
                    if score > best_score:
                        best_score = score
                        candidate = eid
            else:
                # fallback: pick edge incident to highest degree sum
                best_deg = -1
                for eid, (u, v) in edge_id_to_uv.items():
                    if any(pair[1] == eid for pair in work.get(u, set())):
                        deg_sum = len(work.get(u, set())) + len(work.get(v, set()))
                        if deg_sum > best_deg:
                            best_deg = deg_sum
                            candidate = eid
            if candidate is None:
                break
            order.append(candidate)
            self._remove_edge_in_adj(work, candidate, edge_id_to_uv)
        return order

    def _guided_min_search(self, adj, edge_id_to_uv, greedy_order):
        # Try iterative deepening; prioritize edges by greedy order
        work_edges = [eid for eid in greedy_order if eid in edge_id_to_uv]
        remaining = [eid for eid in edge_id_to_uv.keys() if eid not in set(work_edges)]
        priority = work_edges + remaining

        # Quick check: maybe already collapsed
        if self._is_collapsed(adj):
            return 0, []

        # Limit max depth to len(priority)
        max_d = len(priority)
        for d in range(1, max_d + 1):
            found, order = self._dfs_limited(adj, edge_id_to_uv, priority, d)
            if found:
                return d, order
        return max_d, priority

    def _dfs_limited(self, adj, edge_id_to_uv, priority, depth_limit):
        used = set()
        best_order = None

        def backtrack(work_adj, start_idx, depth, chosen):
            nonlocal best_order
            if self._is_collapsed(work_adj):
                best_order = list(chosen)
                return True
            if depth == 0:
                return False
            # pruning: if no edges remain, return
            remaining_edges = self._collect_edges(work_adj)
            if not remaining_edges:
                return self._is_collapsed(work_adj)
            for i in range(start_idx, len(priority)):
                eid = priority[i]
                if eid in used:
                    continue
                # skip if eid already absent in graph
                if not self._edge_present(work_adj, eid):
                    continue
                used.add(eid)
                tmp = self._copy_adj(work_adj)
                self._remove_edge_in_adj(tmp, eid, edge_id_to_uv)
                chosen.append(eid)
                if backtrack(tmp, i + 1, depth - 1, chosen):
                    return True
                chosen.pop()
                used.remove(eid)
            return False

        start = self._copy_adj(adj)
        if backtrack(start, 0, depth_limit, []):
            return True, best_order
        return False, None

    def _edge_present(self, adj, eid):
        for u, neigh in adj.items():
            for v, e in neigh:
                if e == eid:
                    return True
        return False

    def _collect_edges(self, adj):
        s = set()
        for u, neigh in adj.items():
            for v, e in neigh:
                s.add(e)
        return s

    def _get_region_edges(self, network: BaseNetwork, xmin: float, xmax: float, ymin: float, ymax: float) -> set[int]:
        """Get edge IDs that intersect the flush region."""
        region_edges = set()
        for edge in network.get_edges():
            a = network.get_node_by_id(edge.n_from)
            b = network.get_node_by_id(edge.n_to)
            if a is None or b is None:
                continue
            exmin = min(a.n_x, b.n_x)
            exmax = max(a.n_x, b.n_x)
            if not (exmax < xmin or exmin > xmax):
                region_edges.add(int(edge.get_id()))
        return region_edges

    def _compute_region_min_cut(self, network: BaseNetwork, xmin: float, xmax: float, ymin: float, ymax: float) -> list[int]:
        """Compute min-cut for current region using Dinic algorithm."""
        nodes = network.get_nodes()
        if not nodes:
            return []
            
        # Find left and right nodes
        left_nodes = {n.get_id() for n in nodes if n.n_x <= xmin}
        right_nodes = {n.get_id() for n in nodes if n.n_x >= xmax}
        
        if not left_nodes or not right_nodes:
            return []
        
        # Build flow graph
        from collections import defaultdict, deque
        
        def edge_in_region(edge):
            a = network.get_node_by_id(edge.n_from)
            b = network.get_node_by_id(edge.n_to)
            if a is None or b is None:
                return False
            exmin = min(a.n_x, b.n_x)
            exmax = max(a.n_x, b.n_x)
            return not (exmax < xmin or exmin > xmax)
        
        graph = defaultdict(list)
        
        def add_edge(u, v, cap, eid=None):
            graph[u].append([v, cap, len(graph[v]), eid])
            graph[v].append([u, 0, len(graph[u]) - 1, None])
        
        S = 10**9 + 7
        T = 10**9 + 9
        BIG = 10**9
        
        # Add network edges
        for e in network.get_edges():
            u = int(e.n_from)
            v = int(e.n_to)
            if edge_in_region(e):
                cap = 1
            else:
                cap = BIG
            add_edge(u, v, cap, eid=int(e.get_id()))
            add_edge(v, u, cap, eid=int(e.get_id()))
        
        # Connect to source/sink
        for u in left_nodes:
            add_edge(S, int(u), BIG)
        for v in right_nodes:
            add_edge(int(v), T, BIG)
        
        # Dinic max-flow
        def bfs():
            level = {S: 0}
            q = deque([S])
            while q:
                x = q.popleft()
                for y, cap, rev, _ in graph[x]:
                    if cap > 0 and y not in level:
                        level[y] = level[x] + 1
                        q.append(y)
            return level
        
        def dfs(x, T, f, ptr, level):
            if x == T:
                return f
            adj = graph[x]
            i = ptr.get(x, 0)
            while i < len(adj):
                y, cap, rev, _ = adj[i]
                if cap > 0 and level.get(y, -1) == level.get(x, 0) + 1:
                    pushed = dfs(y, T, min(f, cap), ptr, level)
                    if pushed:
                        adj[i][1] -= pushed
                        graph[y][rev][1] += pushed
                        ptr[x] = i
                        return pushed
                i += 1
            ptr[x] = i
            return 0
        
        flow = 0
        while True:
            level = bfs()
            if T not in level:
                break
            ptr = {}
            while True:
                pushed = dfs(S, T, BIG, ptr, level)
                if not pushed:
                    break
                flow += pushed
        
        # Find cut edges
        visited = set()
        stack = [S]
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            for y, cap, _, _ in graph[x]:
                if cap > 0 and y not in visited:
                    stack.append(y)
        
        cut_edges = set()
        for u, adjlist in graph.items():
            if u not in visited:
                continue
            for idx, (v, cap, rev, eid) in enumerate(adjlist):
                rev_edge = graph[v][rev]
                orig_cap = cap + rev_edge[1]
                if v not in visited and orig_cap > 0 and eid is not None and orig_cap < BIG:
                    cut_edges.add(eid)
        
        return list(cut_edges)

    def _compute_min_cut_from_region_edges(self, network: BaseNetwork, xmin: float, xmax: float, 
                                         region_edges: set[int]) -> list[int]:
        """Compute min-cut using only the specified region edges."""
        nodes = network.get_nodes()
        if not nodes:
            return []
            
        # Find left and right nodes
        left_nodes = {n.get_id() for n in nodes if n.n_x <= xmin}
        right_nodes = {n.get_id() for n in nodes if n.n_x >= xmax}
        
        if not left_nodes or not right_nodes:
            return []
        
        # Build flow graph using only region edges
        from collections import defaultdict, deque
        
        graph = defaultdict(list)
        
        def add_edge(u, v, cap, eid=None):
            graph[u].append([v, cap, len(graph[v]), eid])
            graph[v].append([u, 0, len(graph[u]) - 1, None])
        
        S = 10**9 + 7
        T = 10**9 + 9
        BIG = 10**9
        
        # Add only region edges
        for e in network.get_edges():
            eid = int(e.get_id())
            if eid not in region_edges:
                continue
            u = int(e.n_from)
            v = int(e.n_to)
            add_edge(u, v, 1, eid=eid)  # Unit capacity for region edges
            add_edge(v, u, 1, eid=eid)
        
        # Connect to source/sink
        for u in left_nodes:
            add_edge(S, int(u), BIG)
        for v in right_nodes:
            add_edge(int(v), T, BIG)
        
        # Dinic max-flow
        def bfs():
            level = {S: 0}
            q = deque([S])
            while q:
                x = q.popleft()
                for y, cap, rev, _ in graph[x]:
                    if cap > 0 and y not in level:
                        level[y] = level[x] + 1
                        q.append(y)
            return level
        
        def dfs(x, T, f, ptr, level):
            if x == T:
                return f
            adj = graph[x]
            i = ptr.get(x, 0)
            while i < len(adj):
                y, cap, rev, _ = adj[i]
                if cap > 0 and level.get(y, -1) == level.get(x, 0) + 1:
                    pushed = dfs(y, T, min(f, cap), ptr, level)
                    if pushed:
                        adj[i][1] -= pushed
                        graph[y][rev][1] += pushed
                        ptr[x] = i
                        return pushed
                i += 1
            ptr[x] = i
            return 0
        
        flow = 0
        while True:
            level = bfs()
            if T not in level:
                break
            ptr = {}
            while True:
                pushed = dfs(S, T, BIG, ptr, level)
                if not pushed:
                    break
                flow += pushed
        
        # Find cut edges
        visited = set()
        stack = [S]
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            for y, cap, _, _ in graph[x]:
                if cap > 0 and y not in visited:
                    stack.append(y)
        
        cut_edges = set()
        for u, adjlist in graph.items():
            if u not in visited:
                continue
            for idx, (v, cap, rev, eid) in enumerate(adjlist):
                rev_edge = graph[v][rev]
                orig_cap = cap + rev_edge[1]
                if v not in visited and orig_cap > 0 and eid is not None and orig_cap < BIG:
                    cut_edges.add(eid)
        
        return list(cut_edges)

    def _render_initial_flush_region(self, network: BaseNetwork, xmin: float, xmax: float, 
                                   ymin: float, ymax: float, region_edges: set[int], out_path: str):
        """Render initial flush region image before any cuts."""
        nodes = network.get_nodes()
        if not nodes:
            raise ValueError("No nodes to render")
        
        # Image settings
        img_w, img_h = 1200, 800
        margin = 50
        
        # Compute bounds
        x_vals = [n.n_x for n in nodes]
        y_vals = [n.n_y for n in nodes]
        xmin_net, xmax_net = min(x_vals), max(x_vals)
        ymin_net, ymax_net = min(y_vals), max(y_vals)
        
        span_x = max(1e-6, xmax_net - xmin_net)
        span_y = max(1e-6, ymax_net - ymin_net)
        
        def world_to_image(x: float, y: float) -> tuple[int, int]:
            nx = (x - xmin_net) / span_x
            ny = (y - ymin_net) / span_y
            px = int(margin + nx * (img_w - 2 * margin))
            py = int(img_h - (margin + ny * (img_h - 2 * margin)))
            return px, py
        
        # Create image
        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        
        # Draw flush region
        p1 = world_to_image(xmin, ymin)
        p2 = world_to_image(xmax, ymax)
        left = min(p1[0], p2[0])
        right = max(p1[0], p2[0])
        top = min(p1[1], p2[1])
        bottom = max(p1[1], p2[1])
        draw.rectangle([left, top, right, bottom], fill=(255, 0, 0, 40), 
                      outline=(255, 0, 0, 180), width=2)
        
        # Draw edges
        for edge in network.get_edges():
            a = network.get_node_by_id(edge.n_from)
            b = network.get_node_by_id(edge.n_to)
            if a is None or b is None:
                continue
            pa = world_to_image(a.n_x, a.n_y)
            pb = world_to_image(b.n_x, b.n_y)
            eid = int(edge.get_id())
            
            if eid in region_edges:
                color = (40, 90, 200)  # Blue for region edges
                width = 2
            else:
                color = (60, 60, 60)   # Gray for other edges
                width = 1
            
            draw.line([pa, pb], fill=color, width=width)
        
        # Draw nodes
        r = 3
        for n in nodes:
            px, py = world_to_image(n.n_x, n.n_y)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 0, 0))
        
        # Add title
        if font is not None:
            draw.text((10, 10), "Initial Flush Region", fill=(0, 0, 0), font=font)
            draw.text((10, 25), f"Region edges: {len(region_edges)}", fill=(0, 0, 0), font=font)
        
        img.save(out_path)

    def _remove_edge_from_network(self, network: BaseNetwork, edge_id: int):
        """Remove edge from network by ID."""
        network.remove_edge(edge_id)

    def _get_lcc_stats(self, network: BaseNetwork) -> tuple[int, int]:
        """Get largest connected component node and edge counts."""
        edge_id_to_uv, adj = self._build_graph_maps(network)
        max_nodes, max_edges = self._largest_component_stats(adj)
        return max_nodes, max_edges

    def _render_iteration_plot(self, network: BaseNetwork, xmin: float, xmax: float, 
                              ymin: float, ymax: float, region_edges: set[int], 
                              removed_edges: set[int], out_path: str):
        """Render iteration plot with region and removed edges highlighted."""
        nodes = network.get_nodes()
        if not nodes:
            raise ValueError("No nodes to render")
        
        # Image settings
        img_w, img_h = 1200, 800
        margin = 50
        
        # Compute bounds
        x_vals = [n.n_x for n in nodes]
        y_vals = [n.n_y for n in nodes]
        xmin_net, xmax_net = min(x_vals), max(x_vals)
        ymin_net, ymax_net = min(y_vals), max(y_vals)
        
        span_x = max(1e-6, xmax_net - xmin_net)
        span_y = max(1e-6, ymax_net - ymin_net)
        
        def world_to_image(x: float, y: float) -> tuple[int, int]:
            nx = (x - xmin_net) / span_x
            ny = (y - ymin_net) / span_y
            px = int(margin + nx * (img_w - 2 * margin))
            py = int(img_h - (margin + ny * (img_h - 2 * margin)))
            return px, py
        
        # Create image
        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        
        # Draw flush region
        p1 = world_to_image(xmin, ymin)
        p2 = world_to_image(xmax, ymax)
        left = min(p1[0], p2[0])
        right = max(p1[0], p2[0])
        top = min(p1[1], p2[1])
        bottom = max(p1[1], p2[1])
        draw.rectangle([left, top, right, bottom], fill=(255, 0, 0, 40), 
                      outline=(255, 0, 0, 180), width=2)
        
        # Draw edges
        for edge in network.get_edges():
            a = network.get_node_by_id(edge.n_from)
            b = network.get_node_by_id(edge.n_to)
            if a is None or b is None:
                continue
            pa = world_to_image(a.n_x, a.n_y)
            pb = world_to_image(b.n_x, b.n_y)
            eid = int(edge.get_id())
            
            if eid in removed_edges:
                color = (220, 30, 30)
                width = 4
            elif eid in region_edges:
                color = (40, 90, 200)
                width = 2
            else:
                color = (60, 60, 60)
                width = 2
            
            draw.line([pa, pb], fill=color, width=width)
            
            # Label removed edges
            if eid in removed_edges and font is not None:
                mx = (pa[0] + pb[0]) // 2
                my = (pa[1] + pb[1]) // 2
                draw.text((mx + 4, my - 8), str(eid), fill=(220, 30, 30), font=font)
        
        # Draw nodes
        r = 3
        for n in nodes:
            px, py = world_to_image(n.n_x, n.n_y)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 0, 0))
        
        img.save(out_path)


