import gurobipy as gp
from gurobipy import GRB
import networkx as nx

import gurobipy as gp
from gurobipy import GRB

class HarmonicsILP:
    """
    Interval model (non-preemptive) Harmonics-style baseline:
    - Each tenant runs one collective continuously.
    - Decide start times S_m and rate scale r_m in [rmin, 1].
    - If two tenants overlap, enforce pairwise link-capacity constraints using demand fractions.
    Objective: minimize sum of completion times (Avg JCT). Optionally add makespan weight.
    """

    def __init__(self, datacenter, tenant_mapping, tenant_flows, path_table,
                 single_flow_size=None, collective=None, verbose=True, rmin=0.1, makespan_weight=0.0, estimation=0):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.path_table = path_table
        self.single_flow_size = single_flow_size
        self.collective = collective
        self.verbose = verbose

        self.rmin = rmin
        self.makespan_weight = makespan_weight
        self.estimation = estimation

        self.model = None

        # outputs
        self.schedule = {}  # m -> (start_time, rate)

        # prepared data
        self.M = sorted(self.tenant_mapping.keys())
        self.L = list(self.datacenter.topology.edges())
        self.cap = {}
        for u, v in self.datacenter.topology.edges():
            c = float(self.datacenter.topology[u][v]["capacity"])
            self.cap[(u, v)] = c
            self.cap[(v, u)] = c # Ensure full duplex lookup

        self.base_dur = {}     # seconds, if tenant runs alone at full rate
        self.demands = {}      # m -> {e -> demand fraction in [0,1]}

        self._prepare_data()
        self._build_model()

    # ---------- helpers ----------
    def _path_edges(self, s, t):
        path = self.path_table.get((s, t))
        if not path:
            return []
        return list(zip(path[:-1], path[1:]))

    def _prepare_data(self):
        """
        Compute:
        - base_dur[m]: Duration of tenant m if running at rate=1.0 (bottleneck limited)
        - load_bps[m][e]: Physical bandwidth usage (Gbps) of tenant m on link e if rate=1.0
        """
        for m in self.M:
            link_vol_bits = {}  # e -> bits

            # accumulate volume on each physical link for this tenant
            for (u_log, v_log, vol) in self.tenant_flows[m]:
                vol_bits = vol * 1e9 
                u_phy = self.tenant_mapping[m][u_log]
                v_phy = self.tenant_mapping[m][v_log]
                if u_phy == v_phy:
                    continue
                for e in self._path_edges(u_phy, v_phy):
                    link_vol_bits[e] = link_vol_bits.get(e, 0.0) + float(vol_bits)

            # base duration if alone: max_e vol_bits / cap_bits_per_sec
            bd = 0.0
            for e, vb in link_vol_bits.items():
                ce = self.cap.get(e, 0.0)
                if ce > 0:
                    te = vb / ce
                    if te > bd:
                        bd = te

            bd = max(bd, 1e-9)
            self.base_dur[m] = bd

            # Physical Load in Gbps on each used link
            dem = {}
            for e, vb in link_vol_bits.items():
                if vb > 0:
                    # vb / bd = bps
                    # divide by 1e9 to get Gbps
                    gbps = (vb / bd) / 1e9
                    dem[e] = gbps
            self.demands[m] = dem # Now stores Gbps

        if self.verbose:
            print("[HarmonicsIntervalILP] base_dur:", self.base_dur)

    def _build_model(self):
        self.model = gp.Model("Harmonics_TimeSlotted_Discrete")
        self.model.Params.OutputFlag = 1 if self.verbose else 0
        
        # 1. Horizon & Time Slots
        makespan_heu = self.estimation
        
        # Optimize dt for resolution
        # We need dt small enough to distinguish between "serial" and "parallel"
        # base_dur is typically ~0.01s. 
        # If dt = base_dur / 10, we have 10 slots per tenant.
        # This is enough resolution.
        min_base_dur = min(self.base_dur.values()) if self.base_dur else 1.0
        self.dt = min_base_dur / 20.0 # High resolution
        
        # Limit dt to avoid too many slots if base_dur is tiny
        # if self.dt < 1e-5: self.dt = 1e-5
        
        # Horizon: enough to cover makespan_heu
        # Safety buffer 1.5x
        self.H = int((makespan_heu * 5) / self.dt) + 10
        self.T_slots = range(self.H)
        
        if self.verbose:
            print(f"Building Time-Slotted Discrete Model: dt={self.dt}s, Horizon={self.H}")

        # 2. Discrete Rate Levels
        import numpy as np
        self.rates = [round(r, 2) for r in np.linspace(1.0, self.rmin, 10)]
        self.K_levels = range(len(self.rates))

        # Precompute durations (in slots) for each tenant at each rate level
        self.dur_slots = {}
        for m in self.M:
            self.dur_slots[m] = {}
            for k in self.K_levels:
                rate = self.rates[k]
                slots_needed = int(self.base_dur[m] / (rate * self.dt)) + 1
                self.dur_slots[m][k] = slots_needed

        # 3. Variables: x[m, t, k]
        self.x = {}
        for m in self.M:
            for t in self.T_slots:
                for k in self.K_levels:
                    if t + self.dur_slots[m][k] > self.H:
                        continue
                    self.x[m, t, k] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{m}_{t}_{k}")

        # 4. Unique Start Constraint
        for m in self.M:
            lhs = gp.quicksum(self.x[m, t, k] for t in self.T_slots for k in self.K_levels if (m,t,k) in self.x)
            self.model.addConstr(lhs == 1, name=f"StartOnce_{m}")

        # 5. Global Capacity Constraints (Gbps)
        # Sum of (load_Gbps * rate) <= Capacity_Gbps
        
        # Pre-group by link
        link_tenants = {}
        for m in self.M:
            for e in self.demands[m]:
                if e not in link_tenants: link_tenants[e] = []
                link_tenants[e].append(m)
        
        for e, tenants in link_tenants.items():
            if len(tenants) < 2: continue # Optimization
            
            cap_gbps = self.cap[e] / 1e9
            
            for tau in self.T_slots:
                terms = []
                for m in tenants:
                    load_gbps = self.demands[m][e]
                    for k in self.K_levels:
                        rate_ratio = self.rates[k]
                        actual_gbps = load_gbps * rate_ratio
                        dur = self.dur_slots[m][k]
                        
                        # Valid start times t that cover tau
                        t_start = max(0, tau - dur + 1)
                        t_end = tau # inclusive
                        
                        for t in range(t_start, t_end + 1):
                            if (m, t, k) in self.x:
                                terms.append(actual_gbps * self.x[m, t, k])
                
                if terms:
                    # Constraint: Sum of Gbps <= Link Capacity (Gbps)
                    self.model.addConstr(gp.quicksum(terms) <= cap_gbps, name=f"Cap_{e}_{tau}")

        # 6. Objective: Min Avg JCT
        # C_m = Sum_{t,k} x[m,t,k] * (t + dur_slots[m][k]) * dt
        self.C = {}
        makespan = self.model.addVar(vtype=GRB.CONTINUOUS, name="makespan")
        
        obj_terms = []
        for m in self.M:
            completion_time = gp.quicksum(
                self.x[m, t, k] * (t + self.dur_slots[m][k]) * self.dt
                for t in self.T_slots for k in self.K_levels if (m,t,k) in self.x
            )
            self.C[m] = completion_time
            obj_terms.append(completion_time)
            self.model.addConstr(makespan >= completion_time)
            
        self.model.setObjective(gp.quicksum(obj_terms) + self.makespan_weight * makespan, GRB.MINIMIZE)

    def solve(self, timelimit=120, mipgap=0.01):
        self.model.setParam("TimeLimit", timelimit)
        self.model.setParam("MIPGap", mipgap)
        self.model.optimize()

        if self.model.SolCount == 0:
            if self.verbose: print("[Harmonics] No solution found.")
            return None

        sched = {}
        for m in self.M:
            # Re-identify bottleneck capacity for this tenant to convert ratio -> bps
            link_vol_bits = {}
            for (u_log, v_log, vol) in self.tenant_flows[m]:
                vol_bits = vol * 1e9
                u_phy = self.tenant_mapping[m][u_log]
                v_phy = self.tenant_mapping[m][v_log]
                if u_phy == v_phy: continue
                for e in self._path_edges(u_phy, v_phy):
                    link_vol_bits[e] = link_vol_bits.get(e, 0.0) + float(vol_bits)
            
            bottleneck_cap = 1.0
            max_dur = -1.0
            for e, vb in link_vol_bits.items():
                ce = self.cap.get(e, 0.0)
                if ce > 0:
                    dur = vb / ce
                    if dur > max_dur:
                        max_dur = dur
                        bottleneck_cap = ce

            # Extract start time and rate
            for t in self.T_slots:
                for k in self.K_levels:
                    if (m, t, k) in self.x and self.x[m, t, k].X > 0.5:
                        start_time = t * self.dt
                        rate_ratio = self.rates[k]
                        
                        # Convert to physical rate (bps)
                        phy_rate = rate_ratio * bottleneck_cap
                        
                        sched[m] = (start_time * 0.98, phy_rate)
                        if self.verbose:
                            end_time = start_time + self.dur_slots[m][k] * self.dt
                            print(f"Tenant {m}: Start={start_time:.4f} End={end_time:.4f} Rate={phy_rate/1e9:.2f} Gbps (Ratio={rate_ratio:.2f})")
                        break
                if m in sched: break
        
        # --- Debug Analysis ---
        if self.verbose:
            print("\n[Harmonics Debug] Analyzing Link Utilization for the Solution:")
            # Reconstruct load on each link at each time slot
            link_usage = {} # e -> {t -> load_gbps}
            
            for m, (start_t_val, phy_rate) in sched.items():
                # Start slot
                # Use round to avoid precision issues
                start_slot = int(round(start_t_val / self.dt))
                # Duration slots (approx)
                # Need to find rate_ratio from sched? 
                # Let's trust self.demands
                
                # Find k for this m
                k_idx = 0
                for t in self.T_slots:
                    for k in self.K_levels:
                        if (m, t, k) in self.x and self.x[m, t, k].X > 0.5:
                            k_idx = k
                            break
                
                rate_val = self.rates[k_idx]
                dur_slots = self.dur_slots[m][k_idx]
                
                for e, load_gbps in self.demands[m].items():
                    actual_load = load_gbps * rate_val
                    if e not in link_usage: link_usage[e] = [0.0] * self.H
                    
                    for t in range(start_slot, min(self.H, start_slot + dur_slots)):
                        link_usage[e][t] += actual_load

            # Print top utilized links
            print(f"{'Link':<30} {'Cap(Gbps)':<10} {'MaxLoad':<10} {'AvgLoad':<10} {'#Tenants'}")
            sorted_links = sorted(link_usage.keys(), key=lambda e: max(link_usage[e]), reverse=True)
            for e in sorted_links[:10]:
                max_l = max(link_usage[e])
                avg_l = sum(link_usage[e]) / len(link_usage[e])
                cap = self.cap[e] / 1e9
                # Count tenants using this link
                users = [m for m in self.M if e in self.demands[m]]
                print(f"{str(e):<30} {cap:<10.2f} {max_l:<10.2f} {avg_l:<10.2f} {len(users)}")
                
                # Detailed breakdown for top violation
                if max_l > cap * 1.01:
                    print(f"  VIOLATION! Breakdown for {e}:")
                    for m in users:
                        dem = self.demands[m][e]
                        # Find rate used by this tenant
                        used_rate = 0.0
                        if m in sched:
                            # Infer rate from sched (approx) or search x
                            # Let's search x again
                            for t in self.T_slots:
                                for k in self.K_levels:
                                    if (m, t, k) in self.x and self.x[m, t, k].X > 0.5:
                                        used_rate = self.rates[k]
                                        break
                        print(f"    Tenant {m}: Demand={dem:.2f} Gbps, RateRatio={used_rate:.2f}, Actual={dem*used_rate:.2f} Gbps")
                
        return sched

class HarmonicsHeuristic:
    def __init__(self, datacenter, tenant_mapping, tenant_flows, path_table, 
                 single_flow_size, collective="allreduce", verbose=True):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.path_table = path_table
        self.single_flow_size = single_flow_size
        self.collective = collective
        self.verbose = verbose

        self.durations = {} # D_m (max duration across all links for SJF sorting)
        self.durations_per_link = {} # m -> {e -> duration_on_e}
        self.link_users = {} # e -> set(tenants)

        self._prepare_data()

    def _prepare_data(self):
        self.M_tenants = sorted(self.tenant_mapping.keys())
        self.L_links = list(self.datacenter.topology.edges())

        # Link Capacity
        self.cap = {}
        for u, v in self.L_links:
            self.cap[(u,v)] = self.datacenter.topology[u][v]['capacity']

        # Calculate Duration D_m for each tenant
        link_users = {e: set() for e in self.L_links}
        self.durations_per_link = {}

        for m in self.M_tenants:
            self.durations_per_link[m] = {}
            # 1. Calculate load on each link
            link_load = {}
            for (u_log, v_log, vol) in self.tenant_flows[m]:
                u_phy = self.tenant_mapping[m][u_log]
                v_phy = self.tenant_mapping[m][v_log]
                if u_phy == v_phy: continue
                path = self.path_table.get((u_phy, v_phy))
                if not path: continue

                for i in range(len(path)-1):
                    e = (path[i], path[i+1])
                    if e not in link_load: link_load[e] = 0.0
                    link_load[e] += (vol / 8.0)
                    link_users[e].add(m)

            # 2. Find max duration
            max_dur = 0.0
            for e, val in link_load.items():
                if e in self.cap and self.cap[e] > 0:
                    d = (val * 1e9 * 8.0) / self.cap[e]
                    self.durations_per_link[m][e] = d
                    if d > max_dur:
                        max_dur = d

            self.durations[m] = max(max_dur, 1e-6) # Avoid 0

        self.link_users = link_users

        if self.verbose:
            print(f"Harmonics Heuristic Prep: {len(self.M_tenants)} tenants.")
            print(f"Durations (Max): {self.durations}")

    def solve(self):
        """
        A greedy heuristic implementing strictly 'Shortest Job First' (SRPT) with Bandwidth Awareness.
        Instead of strict link locking (Binary Blocking), we use a Time-Bucket approach to pack
        tenants based on their bandwidth demands (Fluid Packing).
        
        1. Sort tenants by duration (Shortest First).
        2. Calculate bandwidth demand for each tenant on each link.
        3. Schedule greedily: Find the earliest time interval where ALL required links have sufficient residual bandwidth.
        """
        # 1. Sort tenants by size (SRPT)
        sorted_tenants = sorted(self.M_tenants, key=lambda m: self.durations[m])
        
        # 2. Time Bucket System
        dt = 0.001 # 1ms resolution
        # Initial horizon estimation (will auto-expand)
        # Max Makespan guess: sum of all durations (worst case serial)
        horizon_slots = int(sum(self.durations.values()) / dt) + 2000
        
        # link_timeline[e] = list of floats (0.0 to 1.0) representing usage at each slot
        link_timeline = {}
        
        final_schedule = {} # m -> (start, end)
        
        if self.verbose:
            print(f"Harmonics Bandwidth-Aware Order: {sorted_tenants}")
            
        for m in sorted_tenants:
            # A. Identify links and calculate bandwidth demands
            # BW_demand[m][e] = DurationOnLink[m][e] / Duration[m]
            # This represents the fraction of link capacity required if the tenant runs at its bottleneck speed.
            
            demands = {} # e -> demand_fraction
            my_links = set()
            
            # Helper to add links
            def add_path_links(u, v):
                if u == v: return
                path = self.path_table.get((u, v))
                if path:
                    for i in range(len(path)-1):
                        e = (path[i], path[i+1])
                        my_links.add(e)

            for (u_log, v_log, vol) in self.tenant_flows[m]:
                u_phy = self.tenant_mapping[m][u_log]
                v_phy = self.tenant_mapping[m][v_log]
                add_path_links(u_phy, v_phy)
            
            # Calculate demand per link
            d_m = self.durations[m]
            for e in my_links:
                d_link = self.durations_per_link[m].get(e, 0.0)
                if d_m > 1e-9:
                    demand = d_link / d_m
                    # Only track if demand is significant
                    if demand > 1e-5:
                        demands[e] = demand
            
            # Duration in slots
            duration_slots = int(d_m / dt) + 1
            
            # B. Find Earliest Fit (First Fit)
            start_t = 0
            
            while True:
                fits = True
                
                # Check constraints for all links at this start_t
                for e, dem in demands.items():
                    # Initialize timeline for this link if new
                    if e not in link_timeline:
                        link_timeline[e] = [0.0] * horizon_slots
                    
                    # Ensure timeline is long enough
                    required_len = start_t + duration_slots
                    if required_len > len(link_timeline[e]):
                        extension = [0.0] * (required_len - len(link_timeline[e]) + 1000)
                        link_timeline[e].extend(extension)
                    
                    # Check capacity violation
                    # Optimization: We can check just the start, middle, end, or step? 
                    # For safety, we check all slots. (It's fast enough for <10k slots)
                    # We can use sum() or any() for speed, but loop is explicit.
                    
                    # Heuristic tolerance: 1.05 to allow slight oversubscription (TCP can handle it)
                    capacity_limit = 1.0 
                    
                    for t in range(start_t, start_t + duration_slots):
                        if link_timeline[e][t] + dem > capacity_limit:
                            fits = False
                            break
                    if not fits: break
                
                if fits:
                    # Found valid slot
                    break
                else:
                    # Try next slot
                    # Optimization: Could jump to next available slot, but +1 is safe
                    start_t += 1
            
            # C. Book Resources
            for e, dem in demands.items():
                # We already ensured length in the check phase
                for t in range(start_t, start_t + duration_slots):
                    link_timeline[e][t] += dem
            
            # D. Record Schedule
            start_time = start_t * dt
            end_time = (start_t + duration_slots) * dt
            final_schedule[m] = (start_time, end_time)
            
        # Optional: Print final stats check
        if len(self.M_tenants) > 0:
            avg_jct = sum(c for _, c in final_schedule.values()) / len(self.M_tenants)
            makespan = max(c for _, c in final_schedule.values())
            if self.verbose:
                print(f"Harmonics BW-Aware Solved: AvgJCT={avg_jct:.4f}, Makespan={makespan:.4f}")

        return final_schedule
