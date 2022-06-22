from mants_sim_to_real.envs.Graph_Env import GraphHabitatEnv
from mants_sim_to_real.envs.Env import MultiHabitatEnv
from mants_sim_to_real.utils.config import get_config

import numpy as np

import torch

def make_graph_eval_env(all_args, run_dir):
    env = GraphHabitatEnv(args=all_args,run_dir=run_dir)
    return env

def make_tans_eval_env(all_args, run_dir):
    env = MultiHabitatEnv(args=all_args, build_graph = False, run_dir=run_dir)
    return env

def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")
    # visual params
    parser.add_argument("--render_merge", action='store_false', default=True,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--visualize_input", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    # graph
    
    parser.add_argument('--map_resolution', type=int, default=5)
    
    parser.add_argument('--num_local_steps', type=int, default=25,
                    help="""Number of steps the local can
                        perform between each global instruction""")
    parser.add_argument('--proj_frontier', action='store_true',
                        default=False, help="by default True, restrict goals to frontiers")
    parser.add_argument('--grid_pos', action='store_true',
                        default=False, help="by default True, use grid_pos")
    parser.add_argument('--agent_invariant', action='store_true',
                        default=False, help="by default True, ")
    parser.add_argument('--grid_goal', default=False, action='store_true')
    parser.add_argument('--use_goal', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_local_single_map', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--grid_last_goal', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--add_grid_pos', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_id_embedding', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_pos_embedding', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_intra_attn', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_self_attn', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_single', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_grid_simple', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--cnn_use_transformer', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_share_cnn_model', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--multi_layer_cross_attn', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--invariant_type', type=str, default = "attn_sum", choices = ["attn_sum", "split_attn", "mean", "alter_attn"])             
    parser.add_argument('--attn_depth', default=2, type=int)
    parser.add_argument('--grid_size', default=8, type=int)
    parser.add_argument('--build_graph', default=False, action='store_true')
    parser.add_argument('--add_ghost', default=False, action='store_true')
    parser.add_argument('--use_merge', default=False, action='store_true')
    parser.add_argument('--use_global_goal', default=False, action='store_true')
    parser.add_argument('--cut_ghost', default=False, action='store_true')
    parser.add_argument('--learn_to_build_graph', default=False, action='store_true')
    parser.add_argument('--use_mgnn', default=False, action='store_true')
    parser.add_argument('--dis_gap', default=2, type=float)
    parser.add_argument('--use_all_ghost_add', default=False, action='store_true')
    parser.add_argument('--ghost_node_size', default=12, type=int)
    parser.add_argument('--use_double_matching', default=False, action='store_true')
    parser.add_argument('--action_mask', default=False, action='store_true')
    parser.add_argument('--matching_type', type=str)
    parser.add_argument('--graph_memory_size', default=100, type=int)
    parser.add_argument('--use_tans', default=False, action='store_true')
    parser.add_argument('--use_vo', default=False, action='store_true')
    parser.add_argument('--ft_use_random', default = False, action='store_true')
    parser.add_argument('--ft_clear_radius',type=float, default = 50)
    parser.add_argument('--ft_cluster_radius', type=float, default = 5.0)
    parser.add_argument('--utility_radius', type = int, default = 30)
    
    # image retrieval
    all_args = parser.parse_known_args(args)[0]

    return all_args

def graph_runner_init(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda 
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    run_dir = './test'
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_graph_eval_env(all_args, run_dir)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    
    from mants_sim_to_real.runner.graph_habitat_runner import GraphHabitatRunner as Runner

    runner = Runner(config)
    
    return runner, all_args.num_local_steps

def tans_runner_init(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda 
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    
    run_dir = './test'
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_tans_eval_env(all_args, run_dir)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    
    from mants_sim_to_real.runner.habitat_runner import HabitatRunner as Runner

    runner = Runner(config)
    
    return runner, all_args.num_local_steps