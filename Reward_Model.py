import torch
import torch.nn as nn
import time
import inspect

'''
TODO:
    Following the input (initial_rank), determine the number of layers 
    and the number of neurons per layer (16 is a dummy value).

    Making inputs (will have 5 possible answers).
    This Reward NN_model should have a 95% accuracy rate or higher.
'''

# Determines what DEVICE the neural network will function on in your computer
DEVICE = "cpu"
if torch.cuda.is_available(): DEVICE = "cuda"
# Apple Silicon has "mps", which has a GPU
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): DEVICE = "mps"

class WardMod(nn.Module):

    def __init__(self, init_neurons = 5, neuron_l1 = 16, neuron_l2 = 8, output = 5):
        # instantiate nn.Module to minimize errors
        super(WardMod, self).__init__()
        # construct the neural network; GELU(approximate='none') is ideal for our Linear Regression
        self.fc1 = nn.Linear(init_neurons, neuron_l1)
        self.GELU = nn.GELU(approximate='none')
        self.fc2 = nn.Linear(neuron_l1, neuron_l2)
        self.out = nn.Linear(neuron_l2, output)

    def forward(self, input_layer):

        #Push inputs across the neural network
        input_layer = self.fc1(input_layer)
        input_layer = self.GELU(input_layer)
        input_layer = self.fc2(input_layer)
        input_layer = self.out(input_layer)
        return input_layer
    
    # Fancy stuff that uses an optimized AdamW optimizer
    def config_optim(self, device):
        
        # Separate the weights from the biases by determining if they require the gradient 
        # or if they are 1-Dimensional, and set appropriate scaling factor to prevent overfitting.
        para_dict = {param_n: param for param_n, param in self.named_parameters()}
        para_dict = { param_n: param for param_n, param in para_dict.items() if param.requires_grad }
        Decay_para = [ param for param_n, param in para_dict.items() if param.dim() >= 2]
        nodecay_para = [ param for param_n, param in para_dict.items() if param.dim() < 2]
        optimized_set = [
            {'params': Decay_para, 'weight_decay': 0.1},
            {'params': nodecay_para, 'weight_decay': 0.0}
        ]
        fuse_exist = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fuse_exist and 'cuda' in device
        optimizer = torch.optim.AdamW(optimized_set, lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=used_fused)
        return optimizer
    
# For the purposes of verifying our results, we will set the seed to some random number. 
# We won't be needing this when we use it to train our AI.
torch.manual_seed(41)

# Let x equal 10 different responses, for now
x = 10

# instantiate; Note that what's given is a dummy input
NN_model = WardMod().to(DEVICE) 
init_input = torch.tensor([ 
    [2.0, 4.0, 6.0, 8.0, 10.0], 
    [9.0, 8.0, 5.0, 3.0, 1.0], 
    [2.0, 5.0, 3.0, 8.0, 10.0], 
    [9.0, 6.0, 7.0, 2.0, 3.0], 
    [1.0, 7.0, 5.0, 3.0, 9.0], 
    [3.0, 1.0, 10.0, 4.0, 2.0], 
    [9.0, 8.0, 5.0, 2.0, 3.0], 
    [1.0, 10.0, 7.0, 4.0, 2.0], 
    [4.0, 6.0, 10.0, 9.0, 2.0], 
    [5.0, 1.0, 9.0, 2.0, 4.0] ] , device = DEVICE) 
result = NN_model(init_input)
optimizer = NN_model.config_optim(device = DEVICE)
# For now, simulate 100 tries
# Don't ask, this is a random 2^N power that had to be small
grad_accum_steps = 4

for i in range(100):

    # Check how long it takes to complete 1 step in the total training process
    t_i = time.time()
    # The desired answer (The answer is supposed to rank human preferences)
    actual_answer = torch.tensor( [
        [0.1, 0.3, 0.5, 0.7, 0.9], 
        [0.9, 0.7, 0.4, 0.2, 0.0], 
        [0.1, 0.4, 0.2, 0.7, 0.9], 
        [0.8, 0.5, 0.6, 0.1, 0.2], 
        [0.0, 0.6, 0.4, 0.2, 0.8], 
        [0.2, 0.0, 0.9, 0.3, 0.1], 
        [0.8, 0.7, 0.4, 0.1, 0.2], 
        [0.0, 0.9, 0.6, 0.3, 0.1], 
        [0.3, 0.5, 0.9, 0.8, 0.1], 
        [0.4, 0.0, 0.8, 0.1, 0.3]], device = DEVICE)
    criteria = nn.MSELoss()
    
    # train the NN_model and zero gradients as .backward() uses a += operator on them
    NN_model.train()
    optimizer.zero_grad()
    
    # Reset
    loss_accum = 0.0
    
    # Execute forwards and backwards (backpropagation) passes 4 times
    for sub_i in range(grad_accum_steps):
        # move tensors to DEVICE, convert to bfloat 16, and determine mean of the loss
        init_input, actual_answer = init_input.to(DEVICE), actual_answer.to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            result = NN_model(init_input)
            loss = criteria(result, actual_answer)
        
        # Scale loss by accumulation steps
        loss = loss / grad_accum_steps
        # Accumulate the loss for reporting
        loss_accum += loss.detach().item()
        # backpropagation
        loss.backward()
    
    # advance the optimizer
    optimizer.step()

    # Wait for GPU to finish as the CPU will not wait by default
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_f = time.time()
    del_t = (t_f - t_i) * 1000

    # I am not flooding my terminal so determine loss once in a while
    if i % 10 == 0:
        print(f"Loss: {loss_accum:.6f} || Time to complete 1 step of training: {del_t:.2f}ms ")


'''
Experiment: Neural Network gets ~75% accuracy after 100 tries, but gets 99.999993% accuracy after 10000 tries
'''