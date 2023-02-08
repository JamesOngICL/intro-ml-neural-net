import copy
import numpy as np
import torch.optim
from part2_utils import *


class Regressor:
    def __init__(self, x, nb_epoch=1000, dropout=0.2, hyperparameters=[13, 24, 36, 1]):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Field declarations
        self.column_dict = {'INLAND': 0, '<1H OCEAN': 1, 'NEAR BAY': 2, 'NEAR OCEAN': 3, 'ISLAND': 4}
        self.y_val = None
        self.x_val = None
        self.best = None
        self.col_avg = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.nb_epoch = nb_epoch

        # Generate correct input dimension based on data
        x_pre = self._preprocessor(x, training=True)
        self.hyperparameters = [x_pre[0].shape[1]] + hyperparameters[1:]

        # creating network
        network = []
        for i in range(len(self.hyperparameters) - 2):
            network.append(nn.Linear(self.hyperparameters[i], self.hyperparameters[i + 1]))
            network.append(nn.Tanh())
            network.append(nn.Dropout(p=dropout, inplace=False))

        network.append(nn.Linear(self.hyperparameters[-2], self.hyperparameters[-1]))
        self.net = nn.Sequential(*network)

        # initialize weights
        self.net.apply(init_weights)
        self.net.double()
        set_seeds(0)
        return
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # One hot encoding
        if training:
            enc_df, label_keys = custom_ohe(x['ocean_proximity'])
            self.column_dict = label_keys  # store the key,value pairs in a dict tracking num columns
        else:
            # Do not generate new key dict during testing
            enc_df, label_keys = custom_ohe(x['ocean_proximity'], self.column_dict)

        x = x.join(enc_df)
        x = x.drop(['ocean_proximity'], axis=1)

        if y is not None:
            x = x.join(y)
            # Drops the nan in median house value column as data would be unusable in neural network
            x = x.dropna(subset=['median_house_value'])
            # Extracts a Column of Median House Values from x dataset.

            # Drop duplicate Y column
            y = x['median_house_value']
            x = x.drop(['median_house_value'], axis=1)
            y = y.to_numpy()

        if training:
            avg_by_column = x.mean()
            self.col_avg = avg_by_column

        x = x.fillna(self.col_avg)

        x = x.to_numpy()

        scaled_y = None

        if training:
            # Derived x parameters from training data
            self.min_x = np.nanmin(x, axis=0)
            self.max_x = np.nanmax(x, axis=0)
            for i in range(0, enc_df.shape[1]):
                self.min_x[len(self.min_x) - 1 - i] = 0
                self.max_x[len(self.min_x) - 1 - i] = 1
            if y is not None:
                self.min_y = np.nanmin(y)
                self.max_y = np.nanmax(y)

        # map the values of the prediction to something way smaller.
        # e.g. need to scale between -1 and 1.

        # Applies scaling formula to X: (xi – min(x)) / (max(x) – min(x)) ([0,2])
        scaled_x = np.divide(2 * np.subtract(x, self.min_x), np.subtract(self.max_x, self.min_x))

        # Shift to [-1, 1]
        scaled_x = np.subtract(scaled_x, 1)  # scales y column accordingly.
        scaled_x = np.nan_to_num(scaled_x, False)

        if y is not None:
            # Applies scaling formula to Y: (yi – min(y) / (max(y) – min(y)) ([0,2])
            scaled_y = np.divide(2 * np.subtract(y, self.min_y), np.subtract(self.max_y, self.min_y))

            # Shift to [-1, 1]
            scaled_y = np.subtract(scaled_y, 1)  # scales y column accordingly.

            scaled_y = np.reshape(scaled_y, (len(scaled_y), 1))
        return scaled_x, scaled_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def convert_y_vals(self, y_vals):
        """Input: y_tensor is a tensor of y_values corresponding to the tensor that we produce values for.
        Output: Numpy array of reversed outputs.
        By default mse_flag is False meaning we don't process MSE loss
        """
        #
        y_numpy = y_vals.detach().numpy()
        # Scales y_vals = (y_vals+1)/2 so as to be between 0 and 1
        y_numpy = np.multiply(np.add(1, y_numpy), 0.5)
        y_numpy = np.add(np.multiply(y_numpy, (self.max_y - self.min_y)), self.min_y)
        return y_numpy

    def fit(self, x, y, x_val=None, y_val=None, batch_size=64, lr=0.0003, verbose=True):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
        Returns:
            self {Regressor} -- Trained model.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Process input dataframe, return numpy arrays
        x_scl, y_scl = self._preprocessor(x, y, training=True)

        if x_val is not None and y_val is not None:
            x_val, y_val = self._preprocessor(x_val, y_val, training=False)

            # Turn validation datasets into tensors
            x_val = torch.from_numpy(x_val)
            y_val = torch.from_numpy(y_val)

        # Rebatch with ndarray as necessary
        if batch_size is not None:
            x_scl, y_scl = batch_data(x_scl, y_scl, batch_size=batch_size)

        # Transform dataframe to torch tensor
        x_scl = torch.from_numpy(x_scl)
        y_scl = torch.from_numpy(y_scl)

        # Define optimization parameters
        optimiser = torch.optim.Adam(self.net.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        last = [10 ** 10] * 20
        minloss = 10 ** 10

        # Average loss array
        train_losses = []
        val_losses = []

        for epoch in range(self.nb_epoch):

            # Train on batched data
            train_loss = 0
            batch_losses = [0]

            for i, batch in enumerate(x_scl):
                self.net.train()
                optimiser.zero_grad()
                y_hat = self.net(batch)
                train_gold = y_scl[i, :]
                train_loss = criterion(y_hat, train_gold)
                train_loss.backward()
                optimiser.step()
                batch_losses.append(train_loss.item())

            train_losses.append(sum(batch_losses) / len(batch_losses))

            # Validation loss
            val_loss = None
            if x_val is not None and y_val is not None:
                self.net.eval()
                y_val_hat = self.net(x_val)
                val_loss = criterion(y_val_hat, y_val)
                val_losses.append(val_loss.item())

            if val_loss is not None and val_loss.item() < minloss:
                minloss = val_loss.item()
                self.best = copy.deepcopy(self.net)

            if verbose:
                # Print debug information
                if val_loss is None:
                    print(f"Epoch: {epoch}, Training loss: {train_loss.item()}")
                else:
                    print(f"Epoch: {epoch}, Training loss: {train_loss.item()}, Validation loss: {val_loss.item()}")

            if epoch > 200 and val_loss is not None and sum(val_losses[-200:-100]) < sum(val_losses[-100:]):
                # We ensure that the net is updated with the best possible version
                print(f"Terminated due to validation loss stagnation at {epoch} epochs with Min loss: {minloss}")
                self.net = self.best
                break

        return self
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x_pre, _ = self._preprocessor(x, training=False)
        self.net.eval()
        y_scl = self.net(torch.from_numpy(x_pre))
        y = self.convert_y_vals(y_scl)
        return y
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        x_pre, y_pre = self._preprocessor(x, y=y, training=False)  # Do not forget
        x_pre = torch.from_numpy(x_pre)
        y_pre = torch.from_numpy(y_pre)
        self.net.eval()
        y_scl = self.net(x_pre)
        y_predicted = self.convert_y_vals(y_scl)
        criterion = torch.nn.MSELoss()
        loss = criterion(torch.from_numpy(y_predicted), torch.from_numpy(self.convert_y_vals(y_pre))).item()
        return np.sqrt(loss)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model, path="part2_model.pickle"):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open(path, 'wb') as target:
        pickle.dump(trained_model, target)
    # print(f"\nSaved model in {path}\n")


def load_regressor(path="part2_model.pickle"):
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open(path, 'rb') as target:
        trained_model = pickle.load(target)
    # print(f"\nLoaded model in {path}\n")
    return trained_model


def RegressorHyperParameterSearchSeed(x=None, y=None, iterations=8, max_epochs=1000):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.
    Arguments:
        Add whatever inputs you need.
    Returns:
        The function should return your optimised hyper-parameters.
    """

    # Load and shuffle
    data = pd.read_csv("housing.csv")
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)

    # Split x and y into separate dataframes
    x = data.loc[:, data.columns != 'median_house_value']
    y = data.loc[:, ['median_house_value']]

    # Dataset setup
    x_train, y_train, x_val, y_val, x_test, y_test = partition_inputs(x, y)

    # Default optimizer parameters
    def_dropout = 0.2
    def_learnr = 0.0003
    def_batch = 64
    best_architecture = [13, 24, 36, 1]

    losses = []

    for i in range(iterations):
        set_seeds(i)
        net = Regressor(x_train, hyperparameters=best_architecture, nb_epoch=max_epochs, dropout=def_dropout)
        net.fit(x_train, y_train, x_val=x_val, y_val=y_val, batch_size=def_batch, lr=def_learnr, verbose=True)

        losses.append([net.score(x_val, y_val), net])

    # Score best
    loss_values = [losses[i][0] for i in range(len(losses))]
    score = min(loss_values)
    best_idx = loss_values.index(score)
    best = losses[best_idx][1]

    print(f"Best Seed: {best_idx}, Validation loss (Best of {iterations}): {score}, Full losses: {loss_values}")

    mse = best.score(x_test, y_test)
    print(f"Test set mse: {mse}, rmse: {np.sqrt(mse)}")
    save_regressor(best, "best_model_seed.pickle")


def example_main():
    """
    Example main used to test that there is no runtime error and
    """
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train, verbose=True)
    prediction = regressor.predict(x_train.loc[0:3])
    print(f"prediction: {prediction}")

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

    save_regressor(regressor, "example_main.pickle")


def example_main_with_train_and_test(test_size=0.2):
    """
    Example main extended for separation the data to the test set.

    Args:
        test_size {float}: ratio for the test set

    Returns:
        None
    """
    # Read data
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Split data into training and test datasets
    x_train = x.loc[:math.floor((1 - test_size) * len(x))]
    y_train = y.loc[:math.floor((1 - test_size) * len(y))]
    x_test = x.loc[math.floor((1 - test_size) * len(x)):]
    y_test = y.loc[math.floor((1 - test_size) * len(y)):]

    # Training
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train, verbose=True)
    save_regressor(regressor, "example_main_with_train_and_test.pickle")

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


def main_load_regressor_and_predict_whole_dataset(dataset_path="housing.csv"):
    """
    Loads the Regressor from the pickle file and calculates RMSE score on the dataset from housing.csv

    Returns:
        None
    """
    # Read data
    output_label = "median_house_value"
    data = pd.read_csv(dataset_path)

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Load regressor
    regressor = load_regressor()

    # Calculate error
    error = regressor.score(x, y)
    print("\nRegressor error: {}\n".format(error))


def copy_network_weights_and_biases():
    """
    Copies information from one Regressor saved in the pickle file to another Regressor.

    Returns:
        None
    """
    best_regressor = load_regressor('best_single_final.pickle')
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    new_regressor = Regressor(x)
    new_regressor.column_dict = copy.deepcopy(best_regressor.column_dict)
    new_regressor.net = copy.deepcopy(best_regressor.net)
    new_regressor.x_val = copy.deepcopy(best_regressor.x_val)
    new_regressor.y_val = copy.deepcopy(best_regressor.y_val)
    new_regressor.min_x = copy.deepcopy(best_regressor.min_x)
    new_regressor.min_y = copy.deepcopy(best_regressor.min_y)
    new_regressor.max_x = copy.deepcopy(best_regressor.max_x)
    new_regressor.max_y = copy.deepcopy(best_regressor.max_y)
    new_regressor.best = copy.deepcopy(best_regressor.best)
    new_regressor.col_avg = copy.deepcopy(best_regressor.col_avg)
    new_regressor.nb_epoch = copy.deepcopy(best_regressor.nb_epoch)
    save_regressor(new_regressor)

    # Calculate error
    error = new_regressor.score(x, y)
    print("\nRegressor error: {}\n".format(error))


if __name__ == '__main__':
    #main_load_regressor_and_predict_whole_dataset()
    copy_network_weights_and_biases()
    # RegressorHyperParameterSearchSeed(iterations=16, max_epochs=4000)
    # example_main()
    # example_main_with_train_and_test()
