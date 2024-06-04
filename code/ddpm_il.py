import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

from my_utils import *
from args_parser import *
from core.agent import Agent

from core.dqn import *
from core.ac import *
from core.irl import *
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import math
""" The main entry function for RL """


def main(args):
    if args.il_method is None:
        method_type = "RL"  # means we just do RL with environment's rewards
    else:
        method_type = "IL"

    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        print(colored("Using CUDA.", p_color))
    np.random.seed(args.seed)
    random.seed(args.seed)
    test_cpu = True  # Set to True to avoid moving gym's state to gpu tensor every step during testing.

    env_name = args.env_name
    print(env_name)
    """ Create environment and get environment's info. """
    env = gym.make(env_name)
    env.seed(args.seed)
    env_test = gym.make(env_name)
    env_test.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    """ always normalize env. """
    if np.asscalar(env.action_space.high[0]) != 1:
        from my_utils.my_gym_utils import NormalizeGymWrapper
        env = NormalizeGymWrapper(env)
        env_test = NormalizeGymWrapper(env_test)
        print("Use state-normalized environments.")
    a_bound = np.asscalar(env.action_space.high[0])
    a_low = np.asscalar(env.action_space.low[0])
    assert a_bound == -a_low
    assert a_bound == 1
    print("State dim: %d, action dim: %d, action bound %d" % (state_dim, action_dim, a_bound))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Define Diffusion Training """
    diffusion = Diff(state_dim=state_dim, action_dim=action_dim, args=args)

    """ Set method and hyper parameter in file name"""
    args.c_data = 2
    method_name = args.il_method.upper() + "_" + args.rl_method.upper()
    hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)
    exp_name = "%s-%s-%s_s%d" % (diffusion.traj_name, method_name, hypers, 1)
    model_path = "./results_%s/%s_opt50_models/%s/%s-%s" % (method_type, method_name, env_name, env_name, exp_name)
    print("Running %s" % (colored(method_name, p_color)))
    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

    # writer = SummaryWriter(result_path)

    """ Reset seed again """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Noise Purification """
    if args.action == 'bc' or args.action == 'bc_diff' or args.action == 'bcnd' or args.action == 'dwbc' or args.action == 'bc_filter':
        resume_iter = 1000
        model_path = model_path + '/%s--iteration-%d-c-%d-model.pth' % (args.env_name, resume_iter, args.c_data)

        # args.c_data = 1
        """ Agent for testing in a separated environemnt """
        agent_test = Agent(env_test, render=False, t_max=args.t_max, test_cpu=test_cpu)
    

        policy_net = Policy_Energy(state_dim, action_dim, hidden_size=args.hidden_size).to(device)

        if args.action == 'bc_diff':
            """ Bahavior Cloning Traning """
            diffusion.noise_purification(model_path, t=args.diff_t)
            diffusion.bc(policy_updater.policy_net, bc_step=10000, denoise=True)
        elif args.action == 'bc_filter':
            diffusion.noise_filter(type='gaussian')
            diffusion.bc(policy_updater.policy_net, bc_step=10000, denoise=True)
        elif args.action == 'bc':
            """ Bahavior Cloning Traning """
            diffusion.bc_energy(policy_net, bc_step=10000)
            # diffusion.bc(policy_updater.policy_net, bc_step=10000)
        elif args.action == 'bcnd':
            """ Bahavior Cloning Traning """
            diffusion.bcnd(policy_updater.policy_net, bc_step=10000)
        elif args.action == 'dwbc':
            """ Bahavior Cloning Traning """
            diffusion.dwbc(policy_updater.policy_net, batch_size=64, bc_step=10000)
        else:
            raise NotImplementedError

        # log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
        log_test = agent_test.collect_samples_test(policy=policy_net, max_num_episodes=10)

        result_text = " | [R_te] "
        result_text += t_format("min: %.2f" % log_test['min_reward'], 1) + t_format(
            "max: %.2f" % log_test['max_reward'], 1) \
                       + t_format("Avg: %.2f (%.2f)" % (log_test['avg_reward'], log_test['std_reward']), 2)
        result_text += "| Distance: "
        result_text += t_format(
            " %.2f (%.2f)" % (log_test['distance'], log_test['distance_std']), 2)
        print(result_text)

    """ Train diffusion model """
    if args.action == 'diff':
        diffusion.update_diffusion(logdir=model_path)

    if args.action == 'gail' or args.action == 'wgail' or args.action=='iwil' or args.action == 'icgail' or args.action == 'uid':
        """ Set method and hyper parameter in file name"""
        if args.denoise:
            method_name = args.action.upper() + str(args.noise_level) + "_DENOISE_" + args.rl_method.upper()
        else:
            method_name = args.action.upper() + str(args.noise_level) + "_" + args.rl_method.upper()

        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)
        exp_name = "%s-%s-%s_s%d" % (diffusion.traj_name, method_name, hypers, args.seed)
        """ Set path for result and model files """
        result_path = "./results_AIL_%s/%s/%s/%s-%s" % (method_type, method_name, env_name, env_name, exp_name)
        model_save_path = "./results_AIL_%s/%s_models/%s/%s-%s" % (
        method_type, method_name, env_name, env_name, exp_name)
        pathlib.Path(model_save_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)

        resume_iter = 3000
        model_path = model_path + '/%s--iteration-%d-c-%d-model.pth' % (args.env_name, resume_iter, args.c_data)

        """ Agent for testing in a separated environment """
        # agent_test = Agent(env_test, render=False, t_max=args.t_max, test_cpu=test_cpu)
        policy_updater = TRPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound,
                              encode_dim=0)
        """ Bahavior Cloning Training """
        diffusion.noise_purification(model_path, t=args.diff_t)
        diffusion.gail_train(state_dim, denoise=args.denoise)

        if args.action == 'iwil':
            diffusion.iwil_pre()
        if args.action == 'icgail':
            diffusion.icgail_pre()
        # diffusion.bc(policy_updater.policy_net, bc_step=10000, denoise=True)

        """ Function to update the parameters of value and policy networks"""

        def update_params_g(batch):
            states = torch.FloatTensor(np.stack(batch.state)).to(device)
            next_states = torch.FloatTensor(np.stack(batch.next_state)).to(device)
            masks = torch.FloatTensor(np.stack(batch.mask)).to(device).unsqueeze(-1)

            actions = torch.FloatTensor(np.stack(batch.action)).to(device)

            if method_type == "RL":
                rewards = torch.FloatTensor(np.stack(batch.reward)).to(device).unsqueeze(-1)
                policy_updater.update_policy(states, actions.to(device), next_states, rewards, masks)
            elif method_type == "IL":
                nonlocal d_rewards
                if 'rex' in args.il_method:
                    d_rewards = diffusion.compute_reward(states, actions, masks=masks).detach().data
                else:
                    d_rewards = diffusion.compute_reward(states, actions).detach().data
                policy_updater.update_policy(states, actions, next_states, d_rewards, masks)

        """ Storage and counters """
        memory = Memory(capacity=1000000)  # Memory buffer with 1 million max size.
        step, i_iter, tt_g, tt_d, perform_test = 0, 0, 0, 0, 0
        d_rewards = torch.FloatTensor(1).fill_(0)  ## placeholder
        log_interval = args.max_step // 1000  # 1000 lines in the text files
        save_model_interval = (log_interval * 10)  # * (platform.system() != "Windows")  # do not save model ?
        print("Max steps: %s, Log interval: %s steps, Model interval: %s steps" % \
              (colored(args.max_step, p_color), colored(log_interval, p_color), colored(save_model_interval, p_color)))

        """ Reset seed again """
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        """ Agent for testing in a separated environment """
        agent_test = Agent(env_test, t_max=args.t_max, test_cpu=test_cpu)

        latent_code = None  ## only for infogail
        state = env.reset()
        done = 1
        """ The actual learning loop"""
        for total_step in tqdm(range(0, args.max_step + 1), dynamic_ncols=True):

            """ Save the learned policy model """
            if save_model_interval > 0 and total_step % save_model_interval == 0:
                policy_updater.save_model("%s_policy_T%d.pt" % (model_save_path, total_step))
                if 'rex' not in args.il_method:
                    diffusion.save_model("%s_reward_T%d.pt" % (model_save_path, total_step))

            """ Test the policy before update """
            if total_step % log_interval == 0:
                perform_test = 1

            """ Test learned policy """
            if perform_test:
                log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
                perform_test = 0

            state_var = torch.FloatTensor(state)

            """ take env step """
            action = policy_updater.sample_action(state_var.to(device).unsqueeze(0)).to(device_cpu).detach().numpy()

            next_state, reward, done, _ = env.step(action)

            if step + 1 == args.t_max:
                done = 1
            memory.push(state, action, int(not done), next_state, reward, latent_code)
            state = next_state
            step = step + 1

            """ reset env """
            if done:  # reset
                state = env.reset()
                step = 0
                latent_code = None

            """ Update policy """
            if memory.size() >= args.big_batch_size and done:
                batch = memory.sample()
                if 'rex' not in args.il_method:
                    for i_d in range(0, args.d_step):
                        index = diffusion.index_sampler()
                        t0_d = time.time()
                        if args.action == 'gail':
                             diffusion.update_discriminator_gail(batch=batch, index=index, total_step=total_step)
                        elif args.action == 'wgail':
                            diffusion.update_discriminator_wgail(batch=batch, index=index, total_step=total_step, policy_updater=policy_updater)
                        elif args.action == 'iwil':
                            diffusion.update_discriminator_iwil(batch=batch, index=index, total_step=total_step)
                        elif args.action == 'icgail':
                            diffusion.update_discriminator_icgail(batch=batch, index=index, total_step=total_step)
                        elif args.action == 'uid':
                            diffusion.update_discriminator_uid(batch=batch, index=index, total_step=total_step)                            
                        else:
                            raise NotImplementedError
                        tt_d += time.time() - t0_d

                t0_g = time.time()
                update_params_g(batch=batch)
                tt_g += time.time() - t0_g
                memory.reset()

            """ Print out result to stdout and save it to a text file for plotting """
            if total_step % log_interval == 0:

                result_text = t_format("Step %7d " % (total_step), 0)
                if method_type == "RL":
                    result_text += t_format("(g%2.2f)s" % (tt_g), 1)
                elif method_type == "IL":
                    c_reward_list = d_rewards.to(device_cpu).detach().numpy()
                    result_text += t_format("(g%2.1f+d%2.1f)s" % (tt_g, tt_d), 1)
                    result_text += " | [D] " + t_format("min: %.2f" % np.amin(c_reward_list), 0.5) + t_format(
                        " max: %.2f" % np.amax(c_reward_list), 0.5)

                result_text += " | [R_te] "

                result_text += t_format("min: %.2f" % log_test['min_reward'], 1) + t_format(
                    "max: %.2f" % log_test['max_reward'], 1) \
                               + t_format("Avg: %.2f (%.2f)" % (log_test['avg_reward'], log_test['std_reward']), 2)
                if args.env_id >= 1 and args.env_id <= 10:
                    result_text += "| Distance: "
                    result_text += t_format(
                        " %.2f (%.2f)" % (log_test['distance'], log_test['distance_std']), 2)
                tt_g = 0
                tt_d = 0
                tqdm.write(result_text)
                with open(result_path + ".txt", 'a') as f:
                    print(result_text, file=f)

if __name__ == "__main__":
    args = args_parser()
    main(args)

