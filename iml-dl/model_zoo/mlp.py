import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size=1, output_size=92, hidden_sizes=None,
                 compress_output=False, input_embedding=False,
                 init_bias_last_layer=1, conv_layer=False, conv_kernel_size=3,
                 fixed_kernel=False, even_odd=False):
        super(MLP, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [64]
        self.init_bias_last_layer = init_bias_last_layer
        self.compress_output = compress_output
        if self.compress_output:
            output_size = output_size // 2
        self.even_odd = even_odd

        self.input_embedding = input_embedding
        if self.input_embedding:
            input_size = input_size * 3
            self.embedding = torch.nn.Embedding(36, input_size)

        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        layers.append(torch.nn.ReLU())

        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
        self.mlp_layers =torch.nn.Sequential(*layers)

        self.fixed_kernel = fixed_kernel
        if conv_layer:
            if self.fixed_kernel:
                self.conv_layer = torch.nn.Conv1d(in_channels=1, out_channels=1,
                                                  kernel_size=conv_kernel_size,
                                                  padding='same', bias=False)
                with torch.no_grad():
                    self.conv_layer.weight.fill_(1.0 / conv_kernel_size)
                self.conv_layer.weight.requires_grad = False
            else:
                self.conv_layer = torch.nn.Conv1d(in_channels=1, out_channels=1,
                                              kernel_size=conv_kernel_size,
                                              padding='same')
        else:
            self.conv_layer = None

        self.sigmoid = torch.nn.Sigmoid()

        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                if module == self.mlp_layers[-1]:
                    if not self.conv_layer:
                        module.bias.data.fill_(self.init_bias_last_layer)
                    else:
                        module.bias.data.fill_(0)
                else:
                    module.bias.data.fill_(0)
        if isinstance(module, torch.nn.Conv1d) and not self.fixed_kernel:
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(self.init_bias_last_layer)

    def forward(self, x):
        if self.even_odd:
            x = x.long() % 2

        if self.input_embedding:
            x = self.embedding(x.long())
            x = x.reshape(x.shape[0], -1)

        x = self.mlp_layers(x)

        if self.conv_layer:
            x = x.unsqueeze(1)
            x = self.conv_layer(x)
            x = x.squeeze(1)

        x = self.sigmoid(x)

        if self.compress_output:
            x = x.repeat_interleave(2, dim=1)

        return x


class DirectParameterOptim(torch.nn.Module):
    def __init__(self, num_slices=32, output_size=92, init_fill=2.0,
                 even_odd=False, activation="sigmoid"):
        super(DirectParameterOptim, self).__init__()
        self.even_odd = even_odd
        self.activation = activation

        if self.even_odd:
            self.optimized_array = torch.nn.Parameter(
                torch.full((2, output_size),
                           init_fill)
            )
        else:
            self.optimized_array = torch.nn.Parameter(
                torch.full((num_slices, output_size),
                           init_fill)
            )

    def forward(self, slice_num):
        index = slice_num.long()
        if self.even_odd:
            index = index % 2
        if self.activation == "sigmoid":
            return torch.sigmoid(self.optimized_array[index[:, 0]])
        elif self.activation == "none":
            return torch.clip(self.optimized_array[index[:, 0]], 0, 1)
        else:
            raise ValueError("Activation function not supported")


class BlockMLP(torch.nn.Module):
    def __init__(self, input_size=1, output_size=92, hidden_sizes=None,
                 compress_output=False, input_embedding=False,
                 init_bias_last_layer=1, size_of_blocks=4, shift_blocks=0):
        super(BlockMLP, self).__init__()

        self.size_of_blocks = size_of_blocks
        self.coarse_output_size = output_size // size_of_blocks + 2
        self.output_size = output_size
        self.shift_blocks = shift_blocks % size_of_blocks
        self.mlp = MLP(input_size, self.coarse_output_size, hidden_sizes,
                       compress_output, input_embedding, init_bias_last_layer)


    def forward(self, x):
        coarse_mask = self.mlp(x)
        fine_mask = coarse_mask.repeat_interleave(
            self.size_of_blocks, dim=1
        )[:, self.shift_blocks:self.output_size+self.shift_blocks]

        return fine_mask
