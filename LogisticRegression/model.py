import numpy as np


class LinearModel:
    '''
    Linear model class.
    '''
    def __init__(self, inp_dim: int, out_dim: int = 1) -> None:
        '''
        Args:
            inp_dim: input dimension
            out_dim: output dimension
        
        Attributes:
            W: weight matrix
        '''
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.W = np.random.randn(inp_dim, out_dim) * (1 / np.sqrt(inp_dim))
        self.b = np.zeros(out_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.

        Args:
            x: input data

        Returns:
            scalar output of linear model
        '''
        return x @ self.W + self.b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.
        '''
        return self.forward(x)
    
    def __repr__(self) -> str:
        '''
        Representation of model.
        '''
        return f'LinearModel({self.inp_dim}, {self.out_dim})'


class LogisticRegression(LinearModel):
    def __init__(self, inp_dim: int) -> None:
        '''
        Args:
            inp_dim: input dimension
        '''
        super().__init__(inp_dim, 1)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        '''
        Stable sigmoid function.

        Args:
            x: input data

        Returns:
            sigmoid of input data
        '''
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.

        Args:
            x: input data

        Returns:
            scalar output of logistic regression model
        '''
        return self._sigmoid(x @ self.W + self.b)


class SoftmaxRegression(LinearModel):
    def __init__(self, inp_dim: int, out_dim: int = 10) -> None:
        '''
        Args:
            inp_dim: input dimension
            out_dim: output dimension
        '''
        super().__init__(inp_dim, out_dim)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        '''
        Stable softmax function.

        Args:
            x: input data

        Returns:
            softmax of input data
        '''
        # raise NotImplementedError(
        #     'Implement softmax here (look for stable softmax implementation online)')
        #x = np.clip(x, -500, 500)
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True)) #max substracted to prevent numerical instability
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.

        Args:
            x: input data

        Returns:
            scalar output of softmax regression model
        '''
        # raise NotImplementedError('Implement forward pass here using softmax implemented above')

        # print(x.shape)
        # print(self.W.shape)
        # print(self.b.shape)
        # print((x @ self.W + self.b))
        # print("Shape of W: ",self.W.shape )
        # print("Shape of x: ",x.shape)
        # print("Shpae of b: ",self.b.shape)

        return self._softmax(x @ self.W + self.b)