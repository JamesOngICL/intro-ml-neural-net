import pandas as pd

from part2_house_value_regression import *
from part2_utils import *

"""
def test_convert_y_vals():
    # initializes a 2d tensor of outputs for testing
    init_np = np.array([[1, 3, 5, 7]])
    min_y = np.min(init_np)
    max_y = np.max(init_np)

    # defines a unit test of scaled inputs
    scaled_y = np.divide(2 * np.subtract(init_np, min_y), np.subtract(max_y, min_y))

    # Shift to [-1, 1]
    scaled_y = np.subtract(scaled_y, 1)  # scales y column accordingly.

    # instantiate a test for mse loss
    test_mse_val = np.array([[0.95, 2.8, 5, 7.1]])
    scaled_test_y = np.subtract(np.divide(2 * np.subtract(test_mse_val, min_y), np.subtract(max_y, min_y)), 1)
    scaled_test_y = np.multiply(scaled_test_y, 2)
    t_mse = nn.MSELoss()
    debug_outp = convert_y_vals(torch.from_numpy(scaled_y), min_y, max_y)
    assert debug_outp.all() == np.array([1, 3, 5, 7]).all()
    return debug_outp
"""
def test_preprocessor_robust(file_path):
    print("----Testing_Preprocessor-----")

    # Pandafy dataset
    extract_dataset = pd.read_csv(file_path)  # uses a copied dataset to test program robustness
    last_col = extract_dataset['median_house_value']  # assumes the dropping is later
    extract_dataset = extract_dataset.drop(['median_house_value'], axis=1)
    # Nickname The Y Column as Median House Value to Enable Entering Preprocessor
    y = {'median_house_value': last_col}
    y = pd.DataFrame(y)
    make_regr = Regressor([11, 10, 10, 10, 1])
    # Extract the Input and Output Torch Tensors
    input_data, output_data = make_regr._preprocessor(extract_dataset, y, True)
    assert (torch.max(torch.from_numpy(output_data)) == 1)  # Values are Scaled between -1->1 values
    check_min_axis = np.amin(input_data, 0)
    check_max_axis = np.amax(input_data, 0)
    # Use a for loop to ascertain that all min_vals are -1 and all maxes are 1. Scaling is Good check.
    # for i in range(len(check_min_axis)):
    #     assert check_min_axis[i] == -1, "Val != -1 Min"
    #     assert check_max_axis[i] == 1, "Val != 1 Max"
    i_p_new, o_p_new = make_regr._preprocessor(pd.read_csv('one_row_exception.csv'),None,False)
    print("Reached line 50",i_p_new)
    assert (len(input_data) == len(output_data))

    # print("check val1",val1)

    print("-----preprocessor tests passed------")

    return


def test_one_hot_encoding_number_of_columns():
    data = pd.read_csv("housing_two_labels_ocean_proximity.csv")

    y = data['median_house_value']  # assumes the dropping is later
    y = y.to_frame('median_house_value')
    x = data.drop(['median_house_value'], axis=1)
    columns, unique_dict = custom_ohe(x['ocean_proximity'])
    columns_numpy = columns.to_numpy()
    expected_columns = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])

    if np.array_equal(columns_numpy, expected_columns):
        print("numpy arrays are equal")
    else:
        print("numpy arrays are not equal")


def test_one_hot_encoding_more_ocean_proximity_labels():
    data = pd.read_csv("housing_six_labels_ocean_proximity.csv")

    y = data['median_house_value']  # assumes the dropping is later
    y = y.to_frame('median_house_value')
    x = data.drop(['median_house_value'], axis=1)
    columns, unique_dict = custom_ohe(x['ocean_proximity'])
    columns_numpy = columns.to_numpy()
    expected_columns = np.array([
        [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5, 0.5]
    ])

    if np.array_equal(columns_numpy, expected_columns):
        print("numpy arrays are equal")
    else:
        print("numpy arrays are not equal")

test_preprocessor_robust('housing.csv')

def test_one_hot_encoding():
    output_label = "median_house_value"
    fit_data = pd.read_csv('housing_two_labels_ocean_proximity.csv')
    predict_data = pd.read_csv('housing_three_labels_ocean_proximity.csv')

    # Splitting input and output
    x_train = fit_data.loc[:, fit_data.columns != output_label]
    y_train = fit_data.loc[:, [output_label]]

    regressor = Regressor(x_train, nb_epoch=10)
    regressor.fit(x_train, y_train, batch_size=1)
    x_input = predict_data.loc[:, predict_data.columns != output_label]
    prediction = regressor.predict(x_input)
    print("Prediction: ", prediction)

def example_main_batch_size_1():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch=10)
    regressor.fit(x_train, y_train, batch_size=1)
    x_input = data.loc[0:3, data.columns != output_label]
    prediction = regressor.predict(x_input)
    print("Prediction: ")
    print(prediction)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

    def main_one_run():
        set_seeds(0)

        output_label = "median_house_value"

        # Load and shuffle
        data = pd.read_csv("housing.csv")
        data = data.sample(frac=1).reset_index(drop=True)

        # Split x and y into separate dataframes
        y = data['median_house_value']  # assumes the dropping is later
        y = y.to_frame('median_house_value')
        x = data.drop(['median_house_value'], axis=1)

        # Inintialize regressor
        regressor = Regressor([13, 10, 10, 10, 1], nb_epoch=2000, dropout=0.2)

        # Prepare data partitions
        x_train, y_train, x_val, y_val, x_test, y_test = partition_inputs(x, y)

        # Training
        regressor.fit(x_train, y_train, x_val=x_val, y_val=y_val, batch_size=32, lr=0.0001, verbose=True)
