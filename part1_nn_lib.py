import math
import functools
import numpy as np
import pickle


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        # takes sum of y_targ*log(probability)->all probability vals. 
        return out  # store the target value and the softmax in thing.

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current = x
        sigmoided = 1 / (1 + np.exp(-x))

        return sigmoided

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).
         -> backpropogation and calculate loss wrt parameters and i.p.'s

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # array containing gradient wrt batch_size,n_in
        sig_val = 1 / (1 + np.exp(-self._cache_current))
        return grad_z * sig_val * (1 - sig_val)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def test_sigmoid():
    print("-------Running Tests Sigmoid-------")
    sigLay = SigmoidLayer()
    sigLay.forward(np.array([2, 3]))
    new_val = sigLay.backward(np.array([1, 1]))  # test for 1 D array
    sigLay.forward(np.array([[1, 3, 5], [4, 7, 9]]))  # test 2D arr.
    new_arr_val = sigLay.backward(np.array([[3, 6, 9], [1, 2, 7]]))

    sigLay.forward(np.array([[-3, -1, 2], [9, 0, 2]]))
    test_neg = sigLay.backward(np.array([[1, 2, 6], [-2, 3, 7]]))
    np.testing.assert_almost_equal(new_val,
                                   [0.104994, 0.045176], decimal=6)
    np.testing.assert_almost_equal(np.array([[0.589835, 0.271060, 0.059833],
                                             [0.017663, 0.001820, 0.000864]]), new_arr_val,
                                   6)  # asserting the arrays are same
    np.testing.assert_almost_equal([[0.045177, 0.393224, 0.629962],
                                    [-0.000247, 0.750000, 0.734955]], test_neg, decimal=6)
    print("--------All Sigmoid Tests Ok!----------")


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None

    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current = x
        relu = (x > 0) * x
        return relu
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        relu = self._cache_current
        grad = grad_z * (relu > 0)
        return grad
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def relu_layer_test():
    relu = ReluLayer()
    relu.forward(np.random.rand(2, 10))


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W = xavier_init((n_in, n_out))
        self._b = xavier_init((1, n_out))

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        out = np.matmul(x, self._W)

        # General bias addition for arbitrary batch sizes
        out = np.add(out, self._b)

        # Outputs
        self._cache_current = x
        return out

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._grad_W_current = np.matmul(np.transpose(self._cache_current), grad_z)
        self._grad_b_current = np.matmul(np.ones((1, grad_z.shape[0])), grad_z)
        grad_x = np.matmul(grad_z, np.transpose(self._W))
        return grad_x

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self._b = np.subtract(self._b, np.multiply(learning_rate, self._grad_b_current))
        self._W = np.subtract(self._W, np.multiply(learning_rate, self._grad_W_current))
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def linear_layer_test():
    # Instantiate
    lin = LinearLayer(5, 10)

    # Extract biases
    bias_pre = lin.forward(np.zeros((1, 5)))

    # Extract weights
    weights_adj = lin.forward(np.eye(5, M=5)) - bias_pre

    # Check backprop
    lin.forward(np.random.rand(2, 5))
    lin.backward(np.random.rand(2, 10))
    lin.update_params(0.5)

    # Extract biases
    bias_post = lin.forward(np.zeros((1, 5)))

    # Extract weights
    weights_post = lin.forward(np.eye(5, M=5)) - bias_post

    print("Linear test: Dimensions match")
    return


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
             - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        layers = [LinearLayer(input_dim, self.neurons[0])]
        for layer_index in range(1, len(self.neurons)):
            layers.append(LinearLayer(self.neurons[layer_index - 1], self.neurons[layer_index]))
        self._layers = np.array(layers)
        self._activation_functions = np.array(
            list(map(
                lambda func_name: ReluLayer() if func_name == "relu"
                else (SigmoidLayer() if func_name == "sigmoid" else None), self.activations)))
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        layer_output = x
        for layer_index in range(0, len(self._layers)):
            layer_output = self._layers[layer_index].forward(layer_output)
            if self._activation_functions[layer_index] is not None:
                layer_output = self._activation_functions[layer_index].forward(layer_output)
        return layer_output
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        output_gradient = grad_z
        for layer_index in range(len(self._layers) - 1, -1, -1):
            if self._activation_functions[layer_index] is not None:
                output_gradient = self._activation_functions[layer_index].backward(output_gradient)
            output_gradient = self._layers[layer_index].backward(output_gradient)
        return output_gradient
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
            self,
            network,
            batch_size,
            nb_epoch,
            learning_rate,
            loss_fun,
            shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._loss_layer = MSELossLayer() if self.loss_fun == "mse" else CrossEntropyLossLayer()
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        indices = np.arange(0, len(input_dataset))
        np.random.shuffle(indices)
        return (input_dataset[indices], target_dataset[indices])

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for epoch in range(0, self.nb_epoch):
            error = 0
            if self.shuffle_flag:
                input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)
            input_batches = np.array_split(input_dataset, math.ceil(len(input_dataset) / self.batch_size))
            target_batches = np.array_split(target_dataset, math.ceil(len(target_dataset) / self.batch_size))
            for (input_batch, target_batch) in zip(input_batches, target_batches):
                error += self.eval_loss(input_batch, target_batch)
                grad_z = self._loss_layer.backward()
                self.network.backward(grad_z)
                self.network.update_params(self.learning_rate)
            print(f"training on epoch {epoch} with loss {error}")
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).

        Returns:
            a scalar value -- the loss
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        prediction = self.network.forward(input_dataset)
        return self._loss_layer.forward(prediction, target_dataset)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        max_by_category = np.amax(data, axis=0)
        min_by_category = np.amin(data, axis=0)

        self.scaling_factors = np.divide(1, np.subtract(max_by_category, min_by_category))
        self.biases = np.negative(np.multiply(self.scaling_factors, min_by_category))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Data difference to 1
        scaled_data = np.multiply(data, self.scaling_factors)

        # Data offset 0
        normalized_data = np.add(scaled_data, self.biases)

        return normalized_data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Restore data offset
        scaled_data = np.subtract(data, self.biases)

        # Restore data difference
        raw_data = np.divide(scaled_data, self.scaling_factors)

        return raw_data

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("preprocessor_test_files/iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


def preprocessor_test():
    dat = np.loadtxt("preprocessor_test_files/iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    x_val = x[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)
    # Verify preprocessor constraints

    # Upper bounds of scaled sets 1?
    x_train_pre_max = np.allclose(np.amax(x_train_pre, axis=0), 1)
    x_val_pre_max = np.allclose(np.amax(x_val_pre, axis=0), 1)

    # Lower bounds of scaled sets 0?
    x_train_pre_min = np.allclose(np.amin(x_train_pre, axis=0), 0)
    x_val_pre_min = np.allclose(np.amin(x_val_pre, axis=0), 0)

    # Originally scaled dataset split
    if (x_val_pre_max or x_train_pre_max) and (x_val_pre_min or x_train_pre_min):
        print("Preprocessor test: Mapping range OK")
    else:
        print("Preprocessor test: Mapping range not OK")

    # Reverse mapping
    x_train_rev = prep_input.revert(x_train_pre)
    x_val_rev = prep_input.revert(x_val_pre)

    if np.allclose(x_train, x_train_rev) and np.allclose(x_val, x_val_rev):
        print("Preprocessor test: Mapping reversed successfully")
    else:
        print("Preprocessor test: Mapping reversal failed")


def trainer_test():
    trainer = Trainer(
        network=None,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )
    input_data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    targets = np.array([
        [1],
        [2],
        [3],
        [4]
    ])
    expected_shuffled_inputs = np.array([[7, 8, 9], [10, 11, 12], [4, 5, 6], [1, 2, 3]])
    expected_shuffled_targets = np.array([[3], [4], [2], [1]])
    np.random.seed(0)
    shuffled_input, shuffled_targets = trainer.shuffle(input_data, targets)
    if not np.array_equal(shuffled_input, expected_shuffled_inputs):
        print("inputs were not shuffled correctly")
    if not np.array_equal(shuffled_targets, expected_shuffled_targets):
        print("targets were not shuffled correctly")
    print("Trainer.shuffle() works correctly")


def multi_layer_tutorial_sheet_test():
    input_dim = 2
    neurons = [2, 1]
    activations = ["sigmoid", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)
    net._layers[0]._W = np.array([[0.1, 0.2], [0.3, 0.4]])
    net._layers[0]._b = np.array([[0.2, -0.1]])
    net._layers[1]._W = np.array([[0.5], [0.6]])
    net._layers[1]._b = np.array([[0.1]])
    y_pred = net.forward(np.array([[0.2, -0.4]]))
    y_expected = np.array([0.6296220526751412])

    if np.array_equal(y_pred, y_expected):
        print(f"{y_pred} is not equal to {y_expected}")
    else:
        print("Forward propagation is correct.")

    y_gold = np.array([[1.5]])
    loss_layer = MSELossLayer()
    loss_layer.forward(y_pred, y_gold)
    grad_z = loss_layer.backward()
    net.backward(grad_z)

    obtained_grad_w = net._layers[0]._grad_W_current
    expected_grad_w = np.array([[-0.04341028119501816, -0.051595844999805976], [0.08682056239003633,  0.10319168999961195]])
    if np.array_equal(obtained_grad_w, expected_grad_w):
        print("W gradients are equal.")
    else:
        print("W gradients are not equal.")

if __name__ == "__main__":
    preprocessor_test()
    linear_layer_test()
    relu_layer_test()
    trainer_test()
    multi_layer_tutorial_sheet_test()
    example_main()
    #test_sigmoid()

