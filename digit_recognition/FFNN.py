import torch as tc
from typing import Callable
from torch.utils.data import DataLoader, Dataset
from typing import Callable
from digit_recognition.ImageDataset import ImageDataset
from digit_recognition.images.data import get_img_data, get_image_path
# python -m digit_recognition.FFNN

class ActivationFunction:
    @staticmethod
    def sigmoid(X: tc.Tensor) -> tc.Tensor:
        return 1/(1 + tc.exp(-X))
    
    @staticmethod
    def relu(X: tc.Tensor) -> tc.Tensor:
        return tc.relu(X)
    
    @staticmethod
    def softmax(X: tc.Tensor) -> tc.Tensor:
        max_X = tc.max(X, dim=1, keepdim=True)[0]
        _X = tc.exp(X - max_X)
        return _X / tc.sum(_X, dim=1, keepdim=True)
    
    @staticmethod
    def copy(X: tc.Tensor) -> tc.Tensor:
        return X
    
class LossFunction:
    @staticmethod
    def cross_entropy(
        X: tc.Tensor, 
        Y: tc.Tensor
    ) -> tc.Tensor:
        """Cross-entropy loss function
        Args:
            X (tc.Tensor): (samples, ouputs) = (m, n)
            Y (tc.Tensor): Class indices - (samples, ) = (m,)
        """
        s = X.shape[0]
        eps = 1e-12
        return -tc.sum(tc.log(X[tc.arange(s), Y] + eps))/s
    
    @staticmethod
    def mse(
        X: tc.Tensor, 
        Y: tc.Tensor
    ) -> tc.Tensor:
        """Mean square error loss function

        Args:
            X (tc.Tensor): (samples, ouputs) = (m, n)
            Y (tc.Tensor): (samples, labels) = (m,)
        """
        Y_one_hot = tc.nn.functional.one_hot(Y, num_classes=X.shape[1]).float()
        return tc.mean((X - Y_one_hot)**2)
        
    @staticmethod 
    def softmax_cross_entropy(
        X: tc.Tensor, 
        Y: tc.Tensor
    ):
        """Softmax cross-entropy loss function

        Args:
            X (tc.Tensor): (samples, ouputs) = (m, n)
            Y (tc.Tensor): (samples, labels) = (m,)
        """
        max_X = tc.max(X, dim=1, keepdim=True)[0]
        Z = X - max_X
        sum = tc.sum(tc.exp(Z), dim=1, keepdim=True)
        s = X.shape[0]
        eps = 1e-12
        return -tc.sum(Z[tc.arange(s), Y] - tc.log(sum + eps).flatten()) / s
        
class NeuralNetwork:
    def __init__(
        self, 
        layers: list[tuple[int, int, Callable[[tc.Tensor], tc.Tensor]]], 
        loss_func: Callable[[tc.Tensor], tc.Tensor],
        output_func: Callable[[tc.Tensor], tc.Tensor] = None
    ):
        """

        Args:
            layers (list[tuple[int, int, Callable[[tc.Tensor], tc.Tensor]]]):
            (output_size, input_size, activation_function)
            loss_func (Callable[[tc.Tensor], tc.Tensor]): 
        """
        self._W: list[tc.Tensor] = []
        self._B: list[tc.Tensor] = []
        self._f: list[Callable[[tc.Tensor], tc.Tensor]] = []
        self._loss_func: Callable[[tc.Tensor, tc.Tensor], tc.Tensor] = loss_func
        self._output_func = self._output_func = output_func if output_func is not None else (lambda x: x)
        
        for i in range(len(layers)):
            m, n, f = layers[i]
            
            W = tc.randn((m, n), requires_grad=True, dtype=tc.float64)
            B = tc.zeros((1, m), requires_grad=True, dtype=tc.float64) 
            

            with tc.no_grad():
                W.mul_(tc.sqrt(tc.tensor(2.0 / n)))
                
            self._W.append(W)
            self._B.append(B)
            self._f.append(f)
        
    def train(
        self, 
        train_data: Dataset, 
        batch_size = 64,
        epoch = 10, 
        alpha = 0.1, 
    ):
        """Training function

        Args:
            train_data (tc.Tensor): (sample x feature)
            batch_size (int, optional): _description_. Defaults to 64.
            epoch (int, optional): _description_. Defaults to 10.
            alpha (float, optional): _description_. Defaults to 0.1.
        """
        W = self._W
        B = self._B
        f = self._f
        loss_func = self._loss_func
        L0 = len(W)
        
        for _ in range(epoch):
            batches = DataLoader(train_data, batch_size, True)
            batch: tuple[tc.Tensor, tc.Tensor]
            for batch in batches:
                A_pre, label = batch
                #print(f"A[0] = {A_pre}")
                for i in range(L0):
                    A_pre = f[i](A_pre @ W[i].T + B[i])
                    # print(f"A[{i + 1}] = {A_pre}")
                    
                loss = loss_func(A_pre, label)
                loss.backward()
                print(f"Loss : {loss}")
                with tc.no_grad():
                    for i in range(L0):
                        # print(f"{i} : {W[i].grad}")
                        self._W[i].sub_(alpha * self._W[i].grad) 
                        self._B[i].sub_(alpha * self._B[i].grad)
                        W[i].grad.zero_()
                        B[i].grad.zero_()
            # alpha += 0.00001
                    
    def predict(self, X: tc.Tensor):
        W = self._W
        B = self._B
        f = self._f
        L0 = len(W)
        
        A: tc.Tensor = X
        with tc.no_grad():
            A = X
            for i in range(L0):
                A = f[i](A @ W[i].T + B[i])
        return self._output_func(A)

nn = NeuralNetwork(
    layers=[
        (600, 10000, ActivationFunction.relu),
        (200, 600, ActivationFunction.relu),
        (10, 200, ActivationFunction.copy)
    ], 
    loss_func=LossFunction.softmax_cross_entropy,
    output_func=ActivationFunction.softmax
)

img_dataset = ImageDataset(
    img_data=get_img_data(), 
    img_encoder=ImageDataset.white_black_encoder, 
    flatten=True
)

nn.train(img_dataset,epoch=10, alpha=0.1, batch_size=100)

for i in range(0, 10):
    res = nn.predict(ImageDataset.img_to_tensor(get_image_path(f"{i}", 10), ImageDataset.white_black_encoder).reshape(1, 10000))[0].tolist()
    max_prob = max(res)
    print(f"Number : {res.index(max_prob)} - {max_prob}")
    print(f"Prob : {res}")

