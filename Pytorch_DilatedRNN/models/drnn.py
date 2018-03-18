"""
Created on Mon Mar 12 10:25:30 2018
@author: Lance Zhang

"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import copy


class multi_dRNN_with_dilations(nn.Module):
    
           
    def __init__(self, hidden_structs, dilations, input_dims):
        """
        Inputs:
            hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
            dilations -- a list, each element indicates the dilation of each layer.
            input_dims -- the input dimension.
        """ 
        super(multi_dRNN_with_dilations, self).__init__()
        
        self.hidden_structs = hidden_structs
        self.dilations = dilations
        
        # define cells
        self.cells = []
        lastHiddenDim = -1
        for i, hidden_dims in enumerate(hidden_structs):
            if i == 0:
                cell = nn.LSTMCell(input_dims, hidden_dims)
            else:
                cell = nn.LSTMCell(lastHiddenDim, hidden_dims)
            
            self.add_module("Cell_{}".format(i), cell)
            self.cells.append(cell)
            lastHiddenDim = hidden_dims
    
    
    def init_hidden(self, batch_size, hidden_dim):
        return (autograd.Variable(torch.zeros(batch_size, hidden_dim)),
                autograd.Variable(torch.zeros(batch_size, hidden_dim)))
    
    
    def dRNN(self, cell, inputs, rate, scope='default'):
        """
        This function constructs a layer of dilated RNN.
        Inputs:
            cell -- the dilation operations is implemented independent of the RNN cell.
                In theory, any valid tensorflow rnn cell should work.
            inputs -- the input for the RNN. inputs should be in the form of
                a list of 'n_steps' tenosrs. Each has shape (batch_size, input_dims)
            rate -- the rate here refers to the 'dilations' in the orginal WaveNet paper. 
            scope -- variable scope.
        Outputs:
            outputs -- the outputs from the RNN.
        """
        n_steps = len(inputs)
        batch_size = inputs[0].size()[0]
        hidden_size = cell.hidden_size
        if rate < 0 or rate >= n_steps:
            raise ValueError('The \'rate\' variable needs to be adjusted.')
#         print ("Building layer: %s, input length: %d, dilation rate: %d, input dim: %d." % (scope, n_steps,rate,inputs[0].size()[1]))
        
    
        # make the length of inputs divide 'rate', by using zero-padding
        EVEN = (n_steps % rate) == 0
        if not EVEN:
            # Create a tensor in shape (batch_size, input_dims), which all elements are zero.  
            # This is used for zero padding
            zero_tensor = autograd.Variable(inputs[0].data.new(inputs[0].data.size()).zero_())
            dialated_n_steps = n_steps // rate + 1 # ceiling
#             print ("=====> %d time points need to be padded. " % (dialated_n_steps * rate - n_steps))
#             print ("=====> Input length for sub-RNN: %d" % (dialated_n_steps))
            for i_pad in range(dialated_n_steps * rate - n_steps):
                inputs.append(zero_tensor)
        else:
            dialated_n_steps = n_steps // rate
#             print ("=====> Input length for sub-RNN: %d" % (dialated_n_steps))
    
        # now the length of 'inputs' divide rate
        # reshape it in the format of a list of tensors
        # the length of the list is 'dialated_n_steps' 
        # the shape of each tensor is [batch_size * rate, input_dims] 
        # by stacking tensors that "colored" the same
    
        # Example: 
        # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
        # zero-padding --> [x1, x2, x3, x4, x5, 0]
        # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
        # which the length is the ceiling of n_steps/rate
        dilated_inputs = [torch.cat([inputs[i * rate + j] for j in range(rate)], dim=0) for i in range(dialated_n_steps)]
        
        dilated_outputs = []
        hidden, cstate = self.init_hidden(batch_size*rate, hidden_size)
        for dilated_input in dilated_inputs:
            hidden, cstate = cell(dilated_input, (hidden, cstate))
            dilated_outputs.append(hidden)
    
        # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
        # split each element of the outputs from size [batch_size*rate, input_dims] to 
        # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
        splitted_outputs = [torch.chunk(output, rate, 0)
                            for output in dilated_outputs]
        unrolled_outputs = [output
                            for sublist in splitted_outputs for output in sublist]
        # remove padded zeros
        outputs = unrolled_outputs[:n_steps]
    
        return outputs 
    

    
    def multi_dRNN(self, inputs):
        """
        Inputs:
            inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dims).
        Outputs:
            outputs -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
        """
        assert (len(self.cells) == len(self.dilations))
        x = copy.copy(inputs) 
        for cell, dilation in zip(self.cells, self.dilations):
            scope_name = "multi_dRNN_dilation_%d" % dilation
            x = self.dRNN(cell, x, dilation, scope=scope_name)
            
        return x   
