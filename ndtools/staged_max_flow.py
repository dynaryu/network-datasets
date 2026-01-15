from scipy.optimize import linprog
import numpy as np
import copy
from typing import Dict

def eval_edge_caps( n_caps_dict, edge_caps_dict, edges, e_max_cap, treat_opposite_edges_as_same_comp=True ):
    """
    Evaluate the capacities of the edges based on the capacities of
    the nodes and the capacities of the edges.

    Args:
    n_caps_dict: dict
        Capacities of the nodes.
    edge_caps_dict: dict
        Capacities of the edges. Key is a tuple of two nodes.
    edges: list of tuples
        List of edges.
    e_max_cap: float
        Maximum capacity of the edges.
    treat_opposite_edges_as_same_comp: bool
        If True, the capacities of the opposite edges are treated as the same component.
    
    Returns:
    e_caps: list of floats
        Capacities of the edges used for optimisation.
    """
    e_caps = []

    for e in edges:
        e_caps_all = []

        if e[0][0] in n_caps_dict:
            e_caps_all.append( n_caps_dict[e[0][0]] )

        if e[0][1] in n_caps_dict:
            e_caps_all.append( n_caps_dict[e[0][1]] )

        if e[0] in edge_caps_dict:
            e_caps_all.append( edge_caps_dict[e[0]] )

        if treat_opposite_edges_as_same_comp:
            if (e[0][1], e[0][0]) in edge_caps_dict:
                e_caps_all.append( edge_caps_dict[(e[0][1], e[0][0])] )

        if len(e_caps_all) > 0:
            e_cap = min(e_caps_all)
        else:
            e_cap = e_max_cap

        e_caps.append( float(e_cap) )

    return e_caps

def create_lp(e_cap, nodes, edges, depots):
    """
    Create a linear programming problem for the pipe network optimisation.

    Args:
    e_cap: list of floats
        Capacities of the edges.
    nodes: dict
        Nodes and their positions.
    edges: list of tuples
        Directional edges.
    depots: list of lists of strings
        Depot nodes.

    Returns:
    c: array
        Coefficients of the objective function.
    A_eq: array
        Coefficients of the equality constraints.
    b_eq: array
        Right-hand side of the equality constraints.
    bounds: list of tuples
        Bounds of the variables.
    """

    n_dpts = len(depots)

    # cost
    n_eds = len(edges)
    c = np.zeros((n_eds+1,))

    # decision variable ({x}, u)
    c[-1] = -1  

    # constraints
    A_eq = np.empty(shape=(0, n_eds+1))
    b_eq = np.empty(shape=(0,))

    # constraint: balance in-flows to depot nodes
    # Depot 1
    A_d_ = np.zeros((1, n_eds+1))
    for idx, e in enumerate(edges):
        i, j, m = e[0][0], e[0][1], e[1]
        if i in depots[0] and m == 1:
            A_d_[0, idx] = 1.0
    A_d_[0, -1] = -1.0
    A_eq = np.vstack((A_eq, A_d_))
    b_eq = np.append(b_eq, 0.0)

    # Depot 2, ..., M
    for d_idx in range(1, n_dpts):
        A_d_ = np.zeros((1, n_eds+1))
        for idx, e in enumerate(edges):
            i, j, m = e[0][0], e[0][1], e[1]
            if j in depots[d_idx] and m == d_idx:
                A_d_[0, idx] = 1.0
        A_d_[0, -1] = -1.0
        A_eq = np.vstack((A_eq, A_d_))
        b_eq = np.append(b_eq, 0.0)

    # constraint: balance in- and out-flows of depot nodes,
    # except for the first and the last depots
    for n in nodes:
        if any( n in d for d in depots ):
            if n not in depots[0] + depots[-1]:
                for d_idx in range(1, n_dpts-1):
                    A_n_ = np.zeros((1, n_eds+1))

                    for e_ in range(n_eds):
                        i, j, m = edges[e_][0][0], edges[e_][0][1], edges[e_][1]
                        if j == n and m == d_idx:
                            A_n_[0, e_] = 1.0
                        elif i == n and m == d_idx+1:
                            A_n_[0, e_] = -1.0

                    if any( A_n_[0, :] != 0.0 ):
                        A_eq = np.vstack((A_eq, A_n_))
                        b_eq = np.append(b_eq, 0.0)

    # contraint: balance flows at non-depot nodes
    for n in nodes: 
        if all( n not in d for d in depots ):
            ms_n = []
            for e_ in edges:
                if e_[0][0] == n:
                    ms_n.append( e_[1] )
            ms_n = np.unique(ms_n)

            for d_idx in ms_n:
                    A_n_ = np.zeros((1, n_eds+1))

                    for e_ in edges:
                        i, j, m = e_[0][0], e_[0][1], e_[1]
                        if j == n and m == d_idx:
                            e_idx = edges.index(e_)
                            A_n_[0, e_idx] = 1.0
                        elif i == n and m == d_idx:
                            e_idx = edges.index(e_)
                            A_n_[0, e_idx] = -1.0

                    A_eq = np.vstack((A_eq, A_n_))
                    b_eq = np.append(b_eq, 0.0)

    # Bounds: capacity of the edges
    bounds = []
    for e_ in range(n_eds):
        bounds.append( (0.0, e_cap[e_]) )
    bounds.append( (0.0, np.inf) )

    return c, A_eq, b_eq, bounds

def get_min_surv_comps_st(e_flows, nodes_name_to_list, edges_name_to_list, st_to_cap):
    """
    Get the minimal component states to ensure system survival.
    components not in the dictionary do not affect the system state.

    Args:
    e_flows: list of floats
        Flows on the edges computed by optimisation.
    nodes_name_to_list: dict
        fragile node name: list of indices in the edges list
    edges_name_to_list: dict
        fragile (=all) edge name: list of indices in the edges list
    st_to_cap: dict
        State to capacity mapping.

    Returns:
    min_surv_comps_st_dict: dict
        minimal component states to ensure system survival.
    """
    n_caps_dict = {}
    for n, e_list in nodes_name_to_list.items():
        n_caps_dict[n] = max([e_flows[i] for i in e_list])

    edge_caps_dict = {}
    for e, e_list in edges_name_to_list.items():
        edge_caps_dict[e] = max([e_flows[i] for i in e_list])

    min_surv_comps_st_dict = {}
    for n, cap in n_caps_dict.items():
        if cap > 0:
            min_st = min((k for k, v in st_to_cap[n].items() if v >= cap), default=None)
            min_surv_comps_st_dict[n] = min_st

    for e, cap in edge_caps_dict.items():
        if cap > 0:
            min_st = min((k for k, v in st_to_cap[e].items() if v >= cap), default=None)
            min_surv_comps_st_dict[e] = min_st

    return min_surv_comps_st_dict

def sys_fun(comps_st: Dict, sys_info: Dict):

    """
    The function to evaluate the system state using the staged maximum flow algorim,
    for a given component states (ref: TBC).

    comps_st: dict[component_name: component_state (int)]

    sys_info: dict with keys:
        'capacity': dict[ node_name: dict[state: capacity] ]
        'node_capacities': dict[ node_name: default_capacity ]
        'edges_name_pair': dict[ component_name: (node1, node2) ]
        'edges': list of tuples [ ((node1, node2), mode) ]
        'edge_max_capacity': float
        'treat_opposite_edges_as_same_comp': bool
        'nodes': dict
        'depots': list of lists of strings
        'nodes_to_edges_idx': dict[ node_name: list of indices in edges list ]
        'edges_name_to_list_idx': dict[ component_name: list of indices in edges list ]
        'threshold': float
    """

    capas = sys_info['capacity']

    n_caps = copy.deepcopy(sys_info['node_capacities'])
    e_caps = {}
    for k, v in comps_st.items():            
        if k in n_caps:
            n_caps[k] = capas[k][v]
        else:
            e_caps[sys_info['edges_name_pair'][k]] = capas[k][v]

    e_caps_list = eval_edge_caps( n_caps, e_caps, sys_info['edges'], sys_info['edge_max_capacity'], sys_info['treat_opposite_edges_as_same_comp'] )
    c_, A_eq_, b_eq_, bounds_ = create_lp(e_caps_list, sys_info['nodes'], sys_info['edges'], sys_info['depots'])
    result_ = linprog(c_, A_eq=A_eq_, b_eq=b_eq_, bounds=bounds_)
    flow = result_.x[-1]
    
    if flow > sys_info['threshold']:
        sys_st = 's'
        min_comp_state = get_min_surv_comps_st(
            result_.x[:-1], sys_info['nodes_to_edges_idx'], sys_info['edges_name_to_list_idx'], capas)                

    else:
        sys_st = 'f'
        min_comp_state = None

    return flow, sys_st, min_comp_state