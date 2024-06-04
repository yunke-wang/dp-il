from my_utils import *
from core_nn.nn_irl import *
# from core_nn.nn_old import *
import h5py
from core_nn.nn_diff import Discriminator_dwbc
import torch

class IRL():
    def __init__(self, state_dim, action_dim, args, initialize_net=True, rebuttal=False):
        self.mini_batch_size = args.mini_batch_size
        self.gp_lambda = args.gp_lambda
        self.gp_alpha = args.gp_alpha
        self.gp_center = args.gp_center
        self.gp_lp = args.gp_lp
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = args.gamma

        self.traj_num = 0

        self.rebuttal = rebuttal
        self.load_demo_list(args, verbose=initialize_net)

        if initialize_net:
            self.initilize_nets(args)

    def initilize_nets(self, args):
        self.discrim_net = Discriminator(self.state_dim, self.action_dim, hidden_size=args.hidden_size,
                                         activation=args.activation, clip=args.clip_discriminator).to(device)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate_d)

    def load_demo_list(self, args, verbose=True):

        index_worker = []
        self.index_worker_idx = []
        self.m_return_list = []
        index_start = 0
        expert_state_list, expert_action_list, expert_reward_list, expert_mask_list, worker_id_list = [], [], [], [], []

        if args.c_data == 1:
            k_spec_list = args.noise_level_list
            traj_path = "../imitation_data/TRAJ_h5_D1/%s" % (args.env_name)

            """ trajectory description in result files """
            self.traj_name = "traj_type%d_N%d" % (args.c_data, args.demo_file_size)

            if args.noise_type != "normal":
                self.traj_name += ("_%s" % args.noise_type)
            if args.c_data == 1:
                self.traj_name += ("_%0.2f" % k_spec_list[0])

        elif args.c_data == 2:
            k_spec_list = args.noise_level_list
            traj_path = "../imitation_data/TRAJ_h5_D2/%s" % (args.env_name)

            """ trajectory description in result files """
            self.traj_name = "traj_type%d_N%d" % (args.c_data, args.demo_file_size)
            if args.demo_split_k:
                self.traj_name = "traj_type%dK_N%d" % (args.c_data, args.demo_file_size)
            if args.c_data == 1:
                self.traj_name += ("_%0.2f" % k_spec_list[0])

        elif args.c_data == 3:
            k_spec_list = args.noise_level_list
            traj_path = "../imitation_data/TRAJ_h5_uniform_noise/%s" % (args.env_name)

            """ trajectory description in result files """
            self.traj_name = "traj_type%d_N%d" % (args.c_data, args.demo_file_size)
            if args.demo_split_k:
                self.traj_name = "traj_type%dK_N%d" % (args.c_data, args.demo_file_size)
            args.demo_file_size = 10000
            args.noise_type = "uniform"

        elif args.c_data == 4:
            k_spec_list = args.noise_level_list
            traj_path = "../imitation_data/TRAJ_h5_salt_noise/%s" % (args.env_name)

            """ trajectory description in result files """
            self.traj_name = "traj_type%d_N%d" % (args.c_data, args.demo_file_size)
            if args.demo_split_k:
                self.traj_name = "traj_type%dK_N%d" % (args.c_data, args.demo_file_size)
            args.demo_file_size = 10000
            args.noise_type = "salt_pepper"

        worker_id = 0
        length = 1000

        for k in range(0, len(k_spec_list)):
            k_spec = k_spec_list[k]

            if args.c_data == 1:
                traj_filename = traj_path + (
                        "/%s_TRAJ-N%d_t%d" % (args.env_name, args.demo_file_size, k_spec))
            else:
                traj_filename = traj_path + (
                            "/%s_TRAJ-N%d_%s%0.2f" % (args.env_name, args.demo_file_size, args.noise_type, k_spec))
                # print(traj_filename,args.noise_type)
            if args.traj_deterministic:
                traj_filename += "_det"

            hf = h5py.File(traj_filename + ".h5", 'r')
            expert_mask = hf.get('expert_masks')[:length]
            expert_mask_list += [expert_mask][:length]
            expert_state_list += [hf.get('expert_states')[:length]]
            expert_action_list += [hf.get('expert_actions')[:length]]
            expert_reward_list += [hf.get('expert_rewards')[:length]]
            reward_array = hf.get('expert_rewards')[:length]
            step_num = expert_mask.shape[0]

            ## Set k=n and K=N. Work and pendulum and lunarlander. The results are not included in the paper.
            if not args.demo_split_k:
                worker_id = k
                worker_id_list += [np.ones(expert_mask.shape) * worker_id]
                self.index_worker_idx += [index_start + np.arange(0, step_num)]
                index_start += step_num

            else:
                ## need to loop through demo until mask = 0, then increase the k counter.
                ## find index in expert_mask where value is 0
                zero_mask_idx = np.where(expert_mask == 0)[0]
                prev_idx = -1
                for i in range(0, len(zero_mask_idx)):
                    worker_id_list += [np.ones(zero_mask_idx[i] - prev_idx) * worker_id]
                    self.index_worker_idx += [index_start + np.arange(0, zero_mask_idx[i] - prev_idx)]
                    index_start += zero_mask_idx[i] - prev_idx

                    worker_id = worker_id + 1
                    prev_idx = zero_mask_idx[i]

            traj_num = step_num - np.sum(expert_mask)
            m_return = np.sum(reward_array) / traj_num

            self.m_return_list += [m_return]
            verbose = False
            if verbose:
                print("TRAJ is loaded from %s with traj_num %s, data_size %s steps, and average return %s" % \
                      (colored(traj_filename, p_color), colored(traj_num, p_color),
                       colored(expert_mask.shape[0], p_color), \
                       colored("%.2f" % (m_return), p_color)))

        expert_states = np.concatenate(expert_state_list, axis=0)
        expert_actions = np.concatenate(expert_action_list, axis=0)
        expert_rewards = np.concatenate(expert_reward_list, axis=0)
        expert_masks = np.concatenate(expert_mask_list, axis=0)
        expert_ids = np.concatenate(worker_id_list, axis=0)

        self.real_state_tensor = torch.FloatTensor(expert_states).to(device_cpu)
        self.real_action_tensor = torch.FloatTensor(expert_actions).to(device_cpu)
        self.real_mask_tensor = torch.FloatTensor(expert_masks).to(device_cpu)
        self.real_worker_tensor = torch.LongTensor(expert_ids).to(device_cpu)
        self.data_size = self.real_state_tensor.size(0)
        # self.worker_num = worker_id + 1 # worker_id start at 0?
        self.worker_num = torch.unique(self.real_worker_tensor).size(0)  # much cleaner code
        self.traj_num = self.real_state_tensor.shape[0]
        verbose = False
        if verbose:
            print("Total data pairs: %s, K %s, state dim %s, action dim %s, a min %s, a_max %s" % \
                  (colored(self.real_state_tensor.size(0), p_color), colored(self.worker_num, p_color), \
                   colored(self.real_state_tensor.size(1), p_color), colored(self.real_action_tensor.size(1), p_color),
                   colored(torch.min(self.real_action_tensor).numpy(), p_color),
                   colored(torch.max(self.real_action_tensor).numpy(), p_color) \
                  ))
        # exit()

    def save_model(self, path):
        torch.save(self.discrim_net.state_dict(), path)

    def load_model(self, path):
        self.discrim_net.load_state_dict(torch.load(path, map_location='cpu'))

    def compute_reward(self, states, actions, next_states=None, masks=None):
        return self.discrim_net.get_reward(states, actions)

    def index_sampler(self, offset=0):
        return torch.randperm(self.data_size - offset)[0:self.mini_batch_size].to(device_cpu)

    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        loss_real = x_real.mean()
        loss_fake = x_fake.mean()
        loss = -(loss_real - loss_fake)

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()

        w_dist = (loss_real - loss_fake).cpu().detach().numpy()
        return w_dist, loss.cpu().detach().numpy()

    def gp_regularizer(self, sa_real, sa_fake):
        if self.gp_lambda == 0:
            return 0

        real_data = sa_real.data
        fake_data = sa_fake.data

        if real_data.size(0) < fake_data.size(0):
            idx = np.random.permutation(fake_data.size(0))[0: real_data.size(0)]
            fake_data = fake_data[idx, :]
        else:
            idx = np.random.permutation(real_data.size(0))[0: fake_data.size(0)]
            real_data = real_data[idx, :]

        if self.gp_alpha == "mix":
            alpha = torch.rand(real_data.size(0), 1).expand(real_data.size()).to(device)
            x_hat = alpha * real_data + (1 - alpha) * fake_data
        elif self.gp_alpha == "real":
            x_hat = real_data
        elif self.gp_alpha == "fake":
            x_hat = fake_data

        x_hat_out = self.discrim_net(x_hat.to(device).requires_grad_())
        gradients = torch.autograd.grad(outputs=x_hat_out, inputs=x_hat, \
                                        grad_outputs=torch.ones(x_hat_out.size()).to(device), \
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        if self.gp_lp:
            return (torch.max(0, gradients.norm(2, dim=1) - self.gp_center) ** 2).mean() * self.gp_lambda
        else:
            return ((gradients.norm(2, dim=1) - self.gp_center) ** 2).mean() * self.gp_lambda

    def behavior_cloning(self, policy_net=None, learning_rate=3e-4, bc_step=0):
        if bc_step <= 0 or policy_net is None:
            return

        bc_step_per_epoch = self.data_size / self.mini_batch_size
        bc_epochs = math.ceil(bc_step / bc_step_per_epoch)

        train = data_utils.TensorDataset(self.real_state_tensor.to(device), self.real_action_tensor.to(device))

        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)
        optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            # for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
            for batch_idx, (s_batch, a_batch) in enumerate(train_loader):
                count = count + 1

                action_mean, _, _ = policy_net(s_batch)
                loss = 0.5 * ((action_mean - a_batch) ** 2).mean()  ##

                optimizer_pi_bc.zero_grad()
                loss.backward()
                optimizer_pi_bc.step()

        t1 = time.time()
        print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1 - t0, loss.item()))

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class CULoss(nn.Module):
    def __init__(self, conf, beta, non=False):
        super(CULoss, self).__init__()
        self.loss = nn.SoftMarginLoss()
        self.beta = beta
        self.non = non
        if conf.mean() > 0.5:
            self.UP = True
        else:
            self.UP = False

    def forward(self, conf, labeled, unlabeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))

        if self.UP:
            # conf_risk = torch.mean((1-conf) * (y_conf_neg - y_conf_pos) + (1 - self.beta) * y_conf_pos)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, torch.ones(unlabeled.shape).to(device)))
            neg_risk = torch.mean((1 - conf) * y_conf_neg)
            pos_risk = torch.mean((conf - self.beta) * y_conf_pos) + unlabeled_risk
        else:
            # conf_risk = torch.mean(conf * (y_conf_pos - y_conf_neg) + (1 - self.beta) * y_conf_neg)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, -torch.ones(unlabeled.shape).to(device)))
            pos_risk = torch.mean(conf * y_conf_pos)
            neg_risk = torch.mean((1 - self.beta - conf) * y_conf_neg) + unlabeled_risk
        if self.non:
            objective = torch.clamp(neg_risk, min=0) + torch.clamp(pos_risk, min=0)
        else:
            objective = neg_risk + pos_risk
        return objective


class PNLoss(nn.Module):
    def __init__(self):
        super(PNLoss, self).__init__()
        self.loss = nn.SoftMarginLoss()

    def forward(self, conf, labeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))

        objective = torch.mean(conf * y_conf_pos + (1 - conf) * y_conf_neg)
        return objective


class Classifier(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.d1(torch.tanh(self.fc1(x)))
        x = self.d2(torch.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x


from core.ddpm import GaussianDiffusion, generate_cosine_schedule, generate_linear_schedule
class Diff(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.load_optimal_demo(env_name=args.env_name)
        self.real_tensor = torch.cat((self.real_state_tensor, self.real_action_tensor), dim=1)

        betas = generate_linear_schedule(T=1000, low=1e-4, high=0.02)
        self.diff = GaussianDiffusion(state_dim, action_dim, betas,
                                      ema_decay=0.9999, ema_update_rate=1, ema_start=2000, loss_type='l2').to(device)
        args.diff_learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.diff.parameters(), lr=args.diff_learning_rate)
        self.save_ckpt_frequency = 1000
        self.args = args
        self.diff_epochs = 20000

        if args.noise_level == 1:
            self.idx_pre, self.idx = 3000, 3500
        elif args.noise_level == 2:
            self.idx_pre, self.idx = 2000, 2500
        elif args.noise_level == 3:
            self.idx_pre, self.idx = 1000, 1500
        elif args.noise_level ==4:
            self.idx_pre, self.idx = 1000, 4000
        else:
            raise NotImplementedError

    def noise_purification(self, model_path, t):
        # if not self.diff.ema:
        self.diff.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        self.diff.to(device)

        ''' Perturb Image '''
        self.sub_demo_tensor = self.real_tensor.to(device)
        batch_size = self.sub_demo_tensor.shape[0]
        noise = torch.randn_like(self.sub_demo_tensor).to(device)
    
        t_batch = t * torch.ones(self.sub_demo_tensor.shape[0],).long().to(device)
        perturbed_x = self.diff.perturb_x(self.sub_demo_tensor, t_batch, noise)

        ''' Remove Noise From Demonstrations '''
        self.clean_x = self.diff.denoise(perturbed_x, t, device, use_ema=False)

    def noise_filter(self, type='gaussian'):
        def mean_filter_1d(x, kernel_size):
            padding = kernel_size // 2
            x = F.pad(x, (padding, padding), mode='reflect')
            return F.avg_pool1d(x.unsqueeze(0), kernel_size=kernel_size, stride=1).squeeze(0)

        def gaussian_filter_1d(x, kernel_size, sigma):
            kernel_size = int(2 * sigma + 1)
            padding = kernel_size // 2
            x = x.unsqueeze(1)  
            x = F.pad(x, (padding, padding), mode='reflect')
            kernel = torch.exp(-(torch.arange(kernel_size, dtype=torch.float32) - padding)**2 / (2 * sigma**2))
            kernel /= torch.sum(kernel)  
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            filtered_x = F.conv1d(x.cpu(), kernel, stride=1).squeeze(1)
            return filtered_x.cuda()
        
        def median_filter_1d(x, kernel_size):
            padding = kernel_size // 2
            x = F.pad(x, (padding, padding), mode='reflect')
            output = torch.zeros_like(x)
            for i in range(x.shape[0]):
                output[i] = torch.median(x[i:i+kernel_size], dim=0).values
            return output

        kernel_size = 3
        self.sub_demo_tensor = self.real_tensor.transpose(0, 1).to(device)
        print(self.sub_demo_tensor.shape)
        if type == 'mean':
            self.clean_x = mean_filter_1d(self.sub_demo_tensor, kernel_size).transpose(0, 1)
        elif type == 'gaussian':
            self.clean_x = gaussian_filter_1d(self.sub_demo_tensor, kernel_size, sigma=1.5).transpose(0, 1)
        elif type == 'median':
            self.clean_x = median_filter_1d(self.sub_demo_tensor, kernel_size).transpose(0, 1)
        else:
            raise NotImplementedError


    def bc(self, policy_net=None, learning_rate=3e-4, bc_step=0, denoise=False):
        if bc_step <= 0 or policy_net is None:
            return
        # bc_step = 100000
        bc_step_per_epoch = self.data_size / self.mini_batch_size
        bc_epochs = math.ceil(bc_step / bc_step_per_epoch)

        if denoise:
            self.sub_demo_tensor = self.clean_x[self.idx_pre:self.idx].to(device)
            if self.args.noise_level ==4:
                idx = np.random.choice(self.sub_demo_tensor.shape[0], 500, replace=False)
                self.sub_demo_tensor = self.sub_demo_tensor[idx]
        else:
            self.sub_demo_tensor = self.real_tensor[self.idx_pre:self.idx].to(device)
            self.demo_tensor = torch.cat((self.opt_state_tensor, self.opt_action_tensor), dim=1)[:50]

            if self.args.noise_level ==4:
                idx = np.random.choice(self.sub_demo_tensor.shape[0], 500, replace=False)
                self.sub_demo_tensor = self.sub_demo_tensor[idx]
            self.sub_demo_tensor = torch.cat((self.sub_demo_tensor, self.demo_tensor.to(device)), dim=0)

        train = data_utils.TensorDataset(self.sub_demo_tensor[:, :self.state_dim].to(device),
                                                            self.sub_demo_tensor[:, self.state_dim:].to(device))
        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)
        optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

        count = 0
        # print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            # for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
            for batch_idx, (s_batch, a_batch) in enumerate(train_loader):
                count = count + 1

                action_mean, _, _ = policy_net(s_batch)
                loss = 0.5 * ((action_mean - a_batch) ** 2).mean()  ##

                optimizer_pi_bc.zero_grad()
                loss.backward()
                optimizer_pi_bc.step()

        t1 = time.time()
        print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1 - t0, loss.item()))


    def update_diffusion(self, logdir):

        self.mini_batch_size = 512
        if self.args.optim_demo:
            self.demo_tensor = torch.cat((self.opt_state_tensor, self.opt_action_tensor), dim=1)[:50]
            print('optimal')
        else:
            self.demo_tensor = self.real_tensor[3000:3500]

        if self.demo_tensor.shape[0] <= self.mini_batch_size:
            self.mini_batch_size = self.demo_tensor.shape[0]

        train = data_utils.TensorDataset(self.demo_tensor.to(device))
        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)

        count = 0
        print("Diffusion Training: %d epochs... " % self.diff_epochs)

        for epoch in range(0, self.diff_epochs+1):
            self.diff.train()
            t0 = time.time()
            train_loss = 0

            for batch_idx, (demo_batch) in enumerate(train_loader):
                count = count + 1

                loss = self.diff(demo_batch[0].to(device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.diff.update_ema()

                train_loss += loss.item()

            train_loss = train_loss / (batch_idx+1)
            t1 = time.time()
            print("Diffusion %s epochs (%2.2fs). Final MSE loss %2.5f" % (colored(epoch, p_color), t1 - t0, train_loss))

            if epoch % 500 == 0:
                model_filename = f"{logdir}/{self.args.env_name}--iteration-{epoch}-c-{self.args.c_data}-model.pth"
                # optim_filename = f"results/{self.args.env_name}--iteration-{epoch}-optim.pth"
                torch.save(self.diff.state_dict(), model_filename)
                # torch.save(self.optimizer.state_dict(), optim_filename)

    def load_optimal_demo(self, env_name, verbose=False):
        # 10000 demo
        traj_filename = '../imitation_data/TRAJ_h5/%s/%s_TRAJ-N1000_normal0.00' % (env_name, env_name)

        hf = h5py.File(traj_filename + ".h5", 'r')
        expert_mask = hf.get('expert_masks')
        expert_mask_list, expert_state_list, expert_action_list, expert_reward_list = [], [], [], []
        expert_mask_list += [expert_mask]
        expert_state_list += [hf.get('expert_states')]
        expert_action_list += [hf.get('expert_actions')]
        expert_reward_list += [hf.get('expert_rewards')]
        reward_array = hf.get('expert_rewards')
        step_num = expert_mask.shape[0]

        expert_states = np.concatenate(expert_state_list, axis=0)
        expert_actions = np.concatenate(expert_action_list, axis=0)
        expert_rewards = np.concatenate(expert_reward_list, axis=0)
        expert_masks = np.concatenate(expert_mask_list, axis=0)
        # expert_ids = np.concatenate(worker_id_list, axis=0)

        self.opt_state_tensor = torch.FloatTensor(expert_states).to(device_cpu)
        self.opt_action_tensor = torch.FloatTensor(expert_actions).to(device_cpu)
        self.opt_mask_tensor = torch.FloatTensor(expert_masks).to(device_cpu)
        self.data_size = self.real_state_tensor.size(0)
        # self.worker_num = worker_id + 1 # worker_id start at 0?
        self.worker_num = torch.unique(self.real_worker_tensor).size(0)  # much cleaner code
        self.traj_num = self.real_state_tensor.shape[0]


    def dwbc(self, policy_net, batch_size=64, bc_step=0):
        LOG_PI_NORM_MAX = 10
        LOG_PI_NORM_MIN = -20

        self.eta = 0.5
        self.d_update_num = 100
        self.alpha = 7.5
        EPS = 1e-7
        self.total_it = 500000

        self.discriminator = Discriminator_dwbc(self.state_dim, self.action_dim).to(device)

        optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=1e-4, weight_decay=0.005)

        self.sub_tensor = self.real_tensor[self.idx_pre:self.idx].to(device)
        if self.args.noise_level ==4:
                idx = np.random.choice(self.sub_tensor.shape[0], 500, replace=False)
                self.sub_tensor = self.sub_tensor[idx]

        self.opt_tensor = torch.cat((self.opt_state_tensor, self.opt_action_tensor), dim=1)[:50].to(device)

        for j in range(bc_step):
            # Sample from D_e and D_o
            sub_idx = np.random.choice(self.opt_tensor.shape[0], batch_size, replace=True)
            opt_idx = np.random.choice(self.sub_tensor.shape[0], batch_size, replace=True)

            state_e = self.sub_tensor[opt_idx, :self.state_dim]
            action_e = self.sub_tensor[opt_idx, self.state_dim:]
            state_o = self.opt_tensor[sub_idx, :self.state_dim]
            action_o = self.opt_tensor[sub_idx, self.state_dim:]

            log_pi_e = policy_net.get_log_prob_no_sum(state_e, action_e)
            log_pi_o = policy_net.get_log_prob_no_sum(state_o, action_o)

            # Compute discriminator loss
            log_pi_e_clip = torch.clip(log_pi_e, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
            log_pi_o_clip = torch.clip(log_pi_o, LOG_PI_NORM_MIN, LOG_PI_NORM_MAX)
            log_pi_e_norm = (log_pi_e_clip - LOG_PI_NORM_MIN) / (LOG_PI_NORM_MAX - LOG_PI_NORM_MIN)
            log_pi_o_norm = (log_pi_o_clip - LOG_PI_NORM_MIN) / (LOG_PI_NORM_MAX - LOG_PI_NORM_MIN)
            d_e = self.discriminator(state_e, action_e, log_pi_e_norm.detach())
            d_o = self.discriminator(state_o, action_o, log_pi_o_norm.detach())

            d_loss_e = -torch.log(d_e)
            d_loss_o = -torch.log(1 - d_o) / self.eta + torch.log(1 - d_e)
            d_loss = torch.mean(d_loss_e + d_loss_o)

            # Optimize the discriminator
            if j % self.d_update_num == 0:
                # print(j)
                self.optimizer_discrim.zero_grad()
                d_loss.backward()
                self.optimizer_discrim.step()

            # Compute policy loss
            d_e_clip = torch.squeeze(d_e).detach()
            d_o_clip = torch.squeeze(d_o).detach()
            d_o_clip[d_o_clip < 0.5] = 0.0

            action_e_predict, _, _ = policy_net(state_e)
            action_o_predict, _, _ = policy_net(state_o)
            log_pi_e = -0.5 * (action_e_predict - action_e) ** 2
            log_pi_o = -0.5 * (action_o_predict - action_o) ** 2

            bc_loss = -torch.sum(log_pi_e, 1)
            corr_loss_e = -torch.sum(log_pi_e, 1) * (self.eta / (d_e_clip * (1.0 - d_e_clip)) + 1.0)
            corr_loss_o = -torch.sum(log_pi_o, 1) * (1.0 / (1.0 - d_o_clip) - 1.0)
            
            p_loss = self.alpha * torch.mean(bc_loss) - torch.mean(corr_loss_e) + torch.mean(corr_loss_o)

            # Optimize the policy
            optimizer_pi_bc.zero_grad()
            p_loss.backward()
            optimizer_pi_bc.step()


    def bcnd(self, policy_net=None, learning_rate=3e-4, bc_step=0):
        if bc_step <= 0 or policy_net is None:
            return

        bc_step_per_epoch = self.data_size / self.mini_batch_size
        bc_epochs = math.ceil(bc_step / bc_step_per_epoch)

        self.sub_demo_tensor = self.real_tensor[self.idx_pre:self.idx].to(device)

        if self.args.noise_level == 4:
            idx = np.random.choice(self.sub_demo_tensor.shape[0], 500, replace=False)
            self.sub_demo_tensor = self.sub_demo_tensor[idx]

        self.demo_tensor = torch.cat((self.opt_state_tensor, self.opt_action_tensor), dim=1)[:50]
        print('optimal')
        self.sub_demo_tensor = torch.cat((self.sub_demo_tensor, self.demo_tensor.to(device)), dim=0)

        train = data_utils.TensorDataset(self.sub_demo_tensor[:, :self.state_dim].to(device),
                                                            self.sub_demo_tensor[:, self.state_dim:].to(device))
        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)
        optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            # for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
            for batch_idx, (s_batch, a_batch) in enumerate(train_loader):
                count = count + 1

                action_mean, _, _ = policy_net(s_batch)

                lg_prob = policy_net.get_log_prob(s_batch, a_batch).detach()
                weight = torch.exp(lg_prob).detach()
                loss = 0.5 * (weight * (action_mean - a_batch) ** 2).mean()  ##

                optimizer_pi_bc.zero_grad()
                loss.backward()
                optimizer_pi_bc.step()

        t1 = time.time()
        print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1 - t0, loss.item()))

    # GAIL
    def gail_train(self, state_dim, denoise=True):
        if not denoise:
            self.sub_demo_tensor = self.real_tensor[self.idx_pre:self.idx].to(device)

            if self.args.noise_level ==4:
                idx = np.random.choice(self.sub_demo_tensor.shape[0], 500, replace=False)
                self.sub_demo_tensor = self.sub_demo_tensor[idx]

            self.demo_tensor = torch.cat((self.opt_state_tensor, self.opt_action_tensor), dim=1)[:50]
            print('optimal')
            self.sub_demo_tensor = torch.cat((self.sub_demo_tensor, self.demo_tensor.to(device)), dim=0)
            self.data_size = self.sub_demo_tensor.shape[0]

            self.real_state_tensor = self.sub_demo_tensor[:, :state_dim]
            self.real_action_tensor = self.sub_demo_tensor[:, state_dim:]
        else:
            self.sub_demo_tensor = self.clean_x[self.idx_pre:self.idx].to(device)
            self.demo_tensor = torch.cat((self.opt_state_tensor, self.opt_action_tensor), dim=1)[:50]

            if self.args.noise_level ==4:
                idx = np.random.choice(self.sub_demo_tensor.shape[0], 500, replace=False)
                self.sub_demo_tensor = self.sub_demo_tensor[idx]

            # print('optimal')
            # self.sub_demo_tensor = torch.cat((self.sub_demo_tensor, self.demo_tensor.to(device)), dim=0)
            self.data_size = self.sub_demo_tensor.shape[0]

            self.real_state_tensor = self.sub_demo_tensor[:, :state_dim]
            self.real_action_tensor = self.sub_demo_tensor[:, state_dim:]

        self.label_real = 0
        self.label_fake = 1  # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid

    def compute_reward(self, states, actions, next_states=None, masks=None):
        # return F.logsigmoid(-self.discrim_net.get_reward(states, actions))   # maximize expert label score.
        return -F.logsigmoid(self.discrim_net.get_reward(states, actions))  # maximize expert label score.

    def update_discriminator_gail(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        adversarial_loss = torch.nn.BCEWithLogitsLoss()
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = adversarial_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake)
        loss = loss_real + loss_fake

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()

    def update_discriminator_wgail(self, batch, index, total_step=0, policy_updater=None):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        if total_step <= 4500000 and total_step / 50000 >= 1:
            log_probs_real = policy_updater.policy_net.get_log_prob(self.real_state_tensor.to(device),
                                                                         self.real_action_tensor.to(device)).detach()
            d = self.discrim_net.get_reward(self.real_state_tensor.to(device), self.real_action_tensor.to(device))
            weight = ((1 / (torch.sigmoid(d) + 1e-6) - 1) * torch.exp(log_probs_real)).pow(1 / 10)
            self.weight = (weight - weight.mean()) / weight.std()
            self.weight = torch.sigmoid(weight)


        if total_step > 55000:
            weight_loss = torch.nn.BCEWithLogitsLoss(weight=self.weight[index, :].detach().to(device))
        adversarial_loss = torch.nn.BCEWithLogitsLoss()
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        if total_step > 55000:
            loss_real = weight_loss(x_real, label_real)
            # print(self.weight[:1000].mean().cpu().detach().numpy(), self.weight[1000:].mean().cpu().detach().numpy())
        else:
            loss_real = adversarial_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake)
        loss = loss_real + loss_fake

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
    
    def update_discriminator_uid(self, batch, index, total_step=0, alpha=0.5, d=4):
        self.alpha = 0.7

        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        adversarial_loss = torch.nn.BCEWithLogitsLoss()
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)

        label_fake_real = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)

        loss_fake = - adversarial_loss(x_fake, label_fake) * self.alpha  #  E_{pi}[log(1-D)] * alpha  with D = sigmoid

        loss_real = - adversarial_loss(x_real, label_real)  #  E_{ex}[log D]

        loss_fake_real_label = - adversarial_loss(x_fake, label_fake_real) * self.alpha  #  E_{pi}[log D] * alpha

        self.threshold = loss_real.detach() - loss_fake_real_label.detach()

        loss = - (loss_fake + loss_real - loss_fake_real_label)

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()

    def iwil_pre(self, args=None):
        # define weight
        optim_conf = torch.ones(50,1)
        non_optim_conf = torch.zeros(500,1)
        self.conf = torch.cat((non_optim_conf, optim_conf), dim=0)

        self.label_real = 0
        self.label_fake = 1  # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid

        # def semi_classifier(self, ratio=0.2):
        ratio = 0.2
        num_label = int(ratio * self.real_state_tensor.shape[0])
        p_idx = np.random.permutation(self.real_state_tensor.shape[0])
        state = self.real_state_tensor[p_idx, :]
        action = self.real_action_tensor[p_idx, :]
        conf = self.conf[p_idx, :]

        labeled_state = state[:num_label, :]
        labeled_action = action[:num_label, :]
        unlabeled_state = state[num_label:, :]
        unlabeled_action = action[num_label:, :]

        labeled_traj = torch.cat((labeled_state, labeled_action), dim=1).to(device)
        unlabeled_traj = torch.cat((unlabeled_state, unlabeled_action), dim=1).to(device)

        classifier = Classifier(labeled_state.shape[1] + labeled_action.shape[1], 40).to(device)
        classifier_optim = torch.optim.Adam(classifier.parameters(), 3e-4, amsgrad=True)
        cu_loss = CULoss(conf, beta=1 - ratio, non=True)

        batch = min(128, labeled_traj.shape[0])
        ubatch = int(batch / labeled_traj.shape[0] * unlabeled_traj.shape[0])
        iters = 25000

        print('Start Pre-train the Semi-supervised Classifier.')

        for i in range(iters):
            l_idx = np.random.choice(labeled_traj.shape[0], batch)
            u_idx = np.random.choice(unlabeled_traj.shape[0], ubatch)

            labeled = classifier(Variable(labeled_traj[l_idx, :]))
            unlabeled = classifier(Variable(unlabeled_traj[u_idx, :]))
            smp_conf = Variable(conf[l_idx, :].to(device))

            classifier_optim.zero_grad()
            risk = cu_loss(smp_conf, labeled, unlabeled)
            risk.backward()
            classifier_optim.step()

            if i % 1000 == 0:
                print('Iteration: {}\t cu_loss: {:.3f}'.format(i, risk.data.item()))

        classifier.eval()

        self.conf = torch.sigmoid(classifier(torch.cat((state, action), dim=1)))
        self.conf[:num_label, :] = conf[:num_label, :]

        self.real_state_tensor = state
        self.real_action_tensor = action

        self.conf.to(device)
        print('Confidence Prediction Ended.')

    def update_discriminator_iwil(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)

        adversarial_loss = torch.nn.BCEWithLogitsLoss()

        " Weighted GAIL Loss"
        p_mean = torch.mean(self.conf)
        p_value = self.conf[index, :] / p_mean

        weighted_loss = torch.nn.BCEWithLogitsLoss(weight=p_value.detach().to(device))
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = weighted_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake)
        loss = loss_real + loss_fake

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()

    def icgail_pre(self):
        optim_conf = torch.ones(50,1)
        non_optim_conf = torch.zeros(500,1)
        self.conf = torch.cat((non_optim_conf, optim_conf), dim=0)

        self.label_real = 0
        self.label_fake = 1  # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid

        ratio = 0.2
        num_label = int(ratio * self.conf.shape[0])
        p_idx = np.random.permutation(self.real_state_tensor.shape[0])
        state = self.real_state_tensor[p_idx, :]
        action = self.real_action_tensor[p_idx, :]
        conf = self.conf[p_idx, :]

        labeled_state = state[:num_label, :]
        labeled_action = action[:num_label, :]
        unlabeled_state = state[num_label:, :]
        unlabeled_action = action[num_label:, :]

        self.labeled_conf = conf[:num_label, :]
        self.labeled_traj = torch.cat((labeled_state, labeled_action), dim=1)
        self.unlabeled_traj = torch.cat((unlabeled_state, unlabeled_action), dim=1)

        self.Z = torch.mean(conf[:num_label, :])
        self.Z = max(self.Z, float(0.7))

    def update_discriminator_icgail(self, batch, index, total_step=0):
        idx = np.random.choice(self.unlabeled_traj.shape[0], int(0.8 * np.stack(batch.state).shape[0]))
        unlabeled = self.unlabeled_traj[idx, :].to(device)

        l_idx = np.random.choice(self.labeled_traj.shape[0], int(0.2 * np.stack(batch.state).shape[0]))
        labeled = self.labeled_traj[l_idx, :].to(device)
        labeled_conf = self.labeled_conf[l_idx, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(unlabeled[:, :self.state_dim], unlabeled[:, self.state_dim:])
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)
        x_real_label = self.discrim_net.get_reward(labeled[:, :self.state_dim], labeled[:, self.state_dim:])

        " IC-GAIL "
        adversarial_loss = torch.nn.BCEWithLogitsLoss()
        weighted_loss = torch.nn.BCEWithLogitsLoss(weight=(1 - labeled_conf))

        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        real_fake = Variable(FloatTensor(x_real_label.size(0), 1).fill_(self.label_fake), requires_grad=False).to(
            device)
        loss_real = adversarial_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake) * self.Z
        loss_weight = weighted_loss(x_real_label, real_fake) * (1 - self.Z) / (1 - labeled_conf.mean())
        loss = loss_real + loss_fake + loss_weight

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((unlabeled[:, :self.state_dim], unlabeled[:, self.state_dim:]), 1),
                                    torch.cat((s_fake, a_fake)[:int(0.8 * np.stack(batch.state).shape[0])], 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()