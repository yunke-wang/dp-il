from my_utils import *

class Policy_Gaussian(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(100, 100), activation='tanh', param_std=1, log_std=0, a_bound=1, tanh_mean=1, squash_action=1):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.tanh_mean = tanh_mean
        self.squash_action = squash_action
        self.param_std = param_std 
        self.log_std_min = -20
        self.log_std_max = 2
        
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim

        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        if self.param_std == 1:
            self.log_std_out = nn.Linear(last_dim, action_dim)  # diagonal gaussian
            self.log_std_out.weight.data.mul_(0.1)
            self.log_std_out.bias.data.mul_(0.0)
        elif self.param_std == 0:
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
            self.entropy_const = action_dim * ( 0.5 + 0.5 * torch.log(2 * torch.FloatTensor(1,1).fill_(math.pi))  ).to(device)
        elif self.param_std == -1:  # policy with fixed variance
            self.action_log_std = torch.ones(1, action_dim).to(device) * np.log(0.1)
            self.entropy_const = action_dim * ( 0.5 + 0.5 * torch.log(2 * torch.FloatTensor(1,1).fill_(math.pi))  ).to(device)

        self.a_bound = a_bound
        # assert self.a_bound == 1

        self.is_disc_action = False

        self.zero_mean = torch.FloatTensor(1, action_dim).fill_(0).to(device) 
        self.unit_var = torch.FloatTensor(1, action_dim).fill_(1).to(device)

        self.logprob_holder = torch.FloatTensor(1).to(device_cpu)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        if self.tanh_mean:
            action_mean = torch.tanh(action_mean) * self.a_bound

        if self.param_std == 1:
            action_log_std = self.log_std_out(x) 
            action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        else:
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def sample_full(self, states, symmetric=0):
        action_mean, action_log_std, action_std = self.forward(states)

        epsilon = torch.FloatTensor(action_mean.size()).data.normal_(0, 1).to(device)
        
        action_raw = action_mean + action_std * epsilon
        log_prob = normal_log_density(action_raw, action_mean, action_log_std, action_std)
        
        if self.squash_action == 1:
            action = torch.tanh(action_raw) * self.a_bound 
            log_prob -= torch.log(1 - torch.tanh(action_raw).pow(2) + 1e-8).sum(1, keepdim=True) 
        elif self.squash_action == -1:
            action = torch.clamp(action_raw, min=-self.a_bound, max=self.a_bound)
        elif self.squash_action == 0: 
            action = action_raw

        if symmetric:
            action_sym_raw = action_mean - action_std * epsilon
            log_prob_sym = normal_log_density(action_sym_raw, action_mean, action_log_std, action_std) 

            if self.squash_action == 1:
                action_sym = torch.tanh(action_sym_raw) * self.a_bound
                log_prob_sym -= torch.log(1 - torch.tanh(action_sym_raw).pow(2) + 1e-8).sum(1, keepdim=True)
            elif self.squash_action == -1:
                action_sym = torch.clamp(action_sym_raw, min=-self.a_bound, max=self.a_bound)
            elif self.squash_action == 0:
                action_sym = action_sym_raw 

            ## concat them along batch dimension, return tensors with double batch size
            action = torch.cat( (action, action_sym), 0 )
            log_prob = torch.cat( (log_prob, log_prob_sym), 0 )

        return action, log_prob, action_mean, action_log_std

    def sample_action(self, x, get_log_prob=False):
        action_mean, action_log_std, action_std = self.forward(x)
        action_raw = torch.normal(action_mean, action_std.to(action_mean.device))
        if self.squash_action == 1:
            action = torch.tanh(action_raw) * self.a_bound
        elif self.squash_action == -1:
            action = torch.clamp(action_raw, min=-self.a_bound, max=self.a_bound)
        elif self.squash_action == 0:
            action = action_raw 

        log_prob = self.logprob_holder.data

        if get_log_prob:
            log_prob = normal_log_density(action_raw, action_mean, action_log_std, action_std)
            if self.squash_action == 1:
                log_prob -= torch.log(1 - torch.tanh(action_raw).pow(2) + 1e-8).sum(1, keepdim=True)

        return action.data.view(-1)

    def sample_action_batched(self, x, get_log_prob=False):
        action_mean, action_log_std, action_std = self.forward(x)
        action_raw = torch.normal(action_mean, action_std.to(action_mean.device))
        if self.squash_action == 1:
            action = torch.tanh(action_raw) * self.a_bound
        elif self.squash_action == -1:
            action = torch.clamp(action_raw, min=-self.a_bound, max=self.a_bound)
        elif self.squash_action == 0:
            action = action_raw

        log_prob = self.logprob_holder.data

        if get_log_prob:
            log_prob = normal_log_density(action_raw, action_mean, action_log_std, action_std)
            if self.squash_action == 1:
                log_prob -= torch.log(1 - torch.tanh(action_raw).pow(2) + 1e-8).sum(1, keepdim=True)

        return action
        
    def greedy_action(self, x):
        action_mean, _, _ = self.forward(x)
        if self.squash_action == 1:
            action_mean = torch.tanh(action_mean) * self.a_bound
        return action_mean.data.view(-1)

    def get_log_prob(self, x, actions_raw, get_log_std=False): 
        action_mean, action_log_std, action_std = self.forward(x)
        log_prob = normal_log_density(actions_raw, action_mean, action_log_std, action_std)
        if self.squash_action:
            log_prob -= torch.log(1 - torch.tanh(actions_raw).pow(2) + 1e-8).sum(1, keepdim=True)
        if get_log_std:
            return log_prob, action_log_std
        else:
            return log_prob

    def get_log_prob_no_sum(self, x, actions_raw, get_log_std=False):
        action_mean, action_log_std, action_std = self.forward(x)
        log_prob = normal_log_density_no_sum(actions_raw, action_mean, action_log_std, action_std)
        return log_prob

    """ Uses for TRPO update with param_std = 0 """
    def compute_entropy(self):
        entropy = self.entropy_const + self.action_log_std.sum()
        return entropy

    """ Only works when param_std = 0 """
    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.data.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.data.view(-1).shape[0]
            id += 1
        return cov_inv, mean, {'std_id': std_id, 'std_index': std_index}

class Value(nn.Module):
    def __init__(self, input_dim, num_outputs=1, hidden_size=(256, 256), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, num_outputs)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = self.value_head(x)
        return x

    def get_value(self, x):
        return self.forward(x) 

""" Two networks in one object"""
class Value_2(nn.Module):
    def __init__(self, input_dim, num_outputs=1, hidden_size=(256, 256), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.affine_layers_1 = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.affine_layers_1.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.value_head_1 = nn.Linear(last_dim, num_outputs)
        self.value_head_1.weight.data.mul_(0.1)
        self.value_head_1.bias.data.mul_(0.0)

        self.affine_layers_2 = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.affine_layers_2.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.value_head_2 = nn.Linear(last_dim, num_outputs)
        self.value_head_2.weight.data.mul_(0.1)
        self.value_head_2.bias.data.mul_(0.0)

    def forward(self, s, a):
        x_1 = torch.cat([s, a], 1)
        for affine in self.affine_layers_1:
            x_1 = self.activation(affine(x_1))
        x_1 = self.value_head_1(x_1)


        x_2 = torch.cat([s, a], 1)
        for affine in self.affine_layers_2:
            x_2 = self.activation(affine(x_2))
        x_2 = self.value_head_2(x_2)

        return x_1, x_2

    def get_q1(self, s, a):
        x_1 = torch.cat([s, a], 1)
        for affine in self.affine_layers_1:
            x_1 = self.activation(affine(x_1))
        x_1 = self.value_head_1(x_1)

        return x_1 


class Policy_Energy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(100, 100), activation='tanh'):
        super().__init__()
        self.activation = F.relu
        
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim + action_dim
        
        self.action_dim = action_dim
        self.state_dim = state_dim

        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.energy = nn.Linear(last_dim, 1)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        energy = self.energy(x)

        return energy

    def info_nce_loss(energy_pos, energy_neg):
        """
        InfoNCE loss function.
        :param energy_pos: Energy of the positive (expert) actions.
        :param energy_neg: Energies of the negative (non-expert) actions.
        :return: The InfoNCE loss.
        """
        numerator = torch.exp(-energy_pos)
        denominator = numerator + torch.exp(-energy_neg).sum(dim=1)
        loss = -torch.log(numerator / denominator).mean()
        return loss

    def batched_loss(self, expert_states, expert_demos, K=16384):
        N, d = expert_demos.shape

        actions_neg = np.random.uniform(-1, 1, (N, K, self.action_dim))

        negative_state = np.repeat(expert_states[:, np.newaxis, :], K, axis=1)

        # print(negative_state.shape, actions_neg.shape)
        # exit()
        negative_samples = np.concatenate((negative_state,actions_neg), axis=2)

        _, K, _ = negative_samples.shape

        # Compute energy for expert demos
        energy_pos = self.forward(torch.Tensor(expert_demos).cuda()).squeeze()

        # Compute energy for negative samples
        energy_neg = self.forward(torch.Tensor(negative_samples).cuda().view(-1, d)).view(N, K)

        # Compute InfoNCE loss
        logits = torch.cat((-energy_pos.unsqueeze(1), -energy_neg), dim=1)  # Combine positive and negative energies
        labels = torch.zeros(N, dtype=torch.long).cuda()  # Labels for expert demos are 0
        loss = F.cross_entropy(logits, labels)

        return loss

    def greedy_action(self,x, iters=3, K=16384,sigma=0.3):
        x = x.cpu().numpy()
        for i in range(iters):
            actions_neg = np.random.uniform(-1, 1, (K, self.action_dim))
            state = np.tile(x, (K, 1))

            candidate_demos = np.concatenate((state, actions_neg),axis=1)
            probabilities = F.softmax(self.forward(torch.Tensor(candidate_demos).cuda()), dim=0).cpu().detach().squeeze(1).numpy()
            # print(probabilities.shape)
            resampled_elements = self.multinomial_resampling(K, probabilities, actions_neg)
            resampled_elements = actions_neg + np.random.normal(loc=0, scale=sigma, size=(K, self.action_dim))
            resampled_elements = np.clip(resampled_elements,-1, 1)

            sigma = sigma * 0.5

        candidate_demos = np.concatenate((state,resampled_elements),axis=1)
        probabilities = F.softmax(self.forward(torch.Tensor(candidate_demos).cuda()), dim=1)
        min_value_index = torch.argmin(probabilities)

        action = candidate_demos[min_value_index, self.action_dim]

        return action

    def multinomial_resampling(self, N_samples, probabilities, elements):
        """
        Perform multinomial resampling based on the given probabilities and elements.
        
        Parameters:
        - N_samples: The number of samples to draw.
        - probabilities: An array of shape (N_samples,) containing the probabilities for each element.
        - elements: An array of shape (N_samples,) containing the elements to be sampled.
        
        Returns:
        - resampled_elements: An array of shape (N_samples,) containing the resampled elements.
        """
        # Ensure probabilities sum to 1
        probabilities /= np.sum(probabilities)
        
        # Draw sample indices based on probabilities
        indices = np.random.choice(np.arange(N_samples), size=N_samples, p=probabilities)
        
        # Select elements based on sampled indices
        resampled_elements = elements[indices]
        
        return resampled_elements
