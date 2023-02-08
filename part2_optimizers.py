from part2_house_value_regression import *
from multiprocessing import Pool


def multi_search(network_architecture, x_train, y_train, x_val, y_val, def_batch, def_max_epochs, def_learnr, dropout,
                 seed):
    set_seeds(seed)
    net = Regressor(network_architecture, nb_epoch=def_max_epochs, dropout=dropout)
    net.fit(x_train, y_train, x_val=x_val, y_val=y_val, batch_size=def_batch, lr=def_learnr, verbose=True)
    return [net.score(x_val, y_val), net]


def RegressorHyperParameterSearch(x, y, threads=8, max_epochs=2000):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.
    Arguments:
        Add whatever inputs you need.
    Returns:
        The function should return your optimised hyper-parameters.
    """
    # Parameter ranges for architectural optimization
    layer1_values = [4]  # range(4, 20, 4)
    layer2_values = range(4, 20, 4)
    layer3_values = range(4, 20, 4)

    # layer1_values = range(10, 20, 4)
    # layer2_values = range(10, 20, 4)
    # layer3_values = range(10, 20, 4)

    # Default optimizer parameters
    def_learnr = 0.0003
    def_batch = 64
    def_dropout = 0.2

    # Dataset setup
    x_train, y_train, x_val, y_val, x_test, y_test = partition_inputs(x, y)

    # Clear files
    # with open("architecture_grid.txt", "w+") as f:
    #     f.write(f"")

    # Search vars
    best_architecture = [13, 10, 10, 10, 1]
    best_loss = 10 ** 10

    # Grid search over layer sizes
    for layer1 in layer1_values:
        for layer2 in layer2_values:
            for layer3 in layer3_values:
                # Construct network object
                network_architecture = [13, layer1, layer2, layer3, 1]

                # Multiprocessing section
                process_args = []

                for i in range(threads):
                    process_args.append(
                        [network_architecture, x_train, y_train, x_val, y_val, def_batch, max_epochs,
                         def_learnr, def_dropout, i])

                with Pool(threads) as p:
                    losses = p.starmap(multi_search, process_args)

                # Score best
                score = min(losses)

                # Write results to file
                print(
                    f"Architecture: {network_architecture}, Validation loss (Best of {threads}): {score}, Full losses: {losses}")
                with open("architecture_grid.txt", "a+") as f:
                    f.write(
                        f"Architecture: {network_architecture}, Validation loss (Best of {threads}): {score}, Full losses: {losses} \n")

                if score < best_loss:
                    best_loss = score
                    best_architecture = network_architecture

    # Write results to file
    print(f"Best architecture: {best_architecture}, Best validation loss: {best_loss}")
    with open("best_params.txt", "a+") as f:
        f.write(f"Best architecture: {best_architecture}, Validation loss: {best_loss} \n")

    return best_architecture


def RegressorHyperParameterSearch2D(x, y, threads=8, max_epochs=2000):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.
    Arguments:
        Add whatever inputs you need.
    Returns:
        The function should return your optimised hyper-parameters.
    """
    # Parameter ranges for architectural optimization
    layer1_values = [36]  # range(32, 40, 4)
    # Matthew
    # layer1_values = [32, 36]

    # Vaclav
    # layer1_values = [24, 28]

    layer2_values = range(4, 40, 4)

    # Default optimizer parameters
    def_learnr = 0.0003
    def_batch = 64
    def_dropout = 0.2

    # Dataset setup
    x_train, y_train, x_val, y_val, x_test, y_test = partition_inputs(x, y)

    # Clear files
    # with open("architecture_grid2D.txt", "w+") as f:
    #     f.write(f"")

    # Search vars
    best_architecture = [13, 10, 10, 1]
    best_loss = 10 ** 10

    # Grid search over layer sizes
    for layer1 in layer1_values:
        for layer2 in layer2_values:
            # Construct network object
            network_architecture = [13, layer1, layer2, 1]

            # Multiprocessing section
            process_args = []

            for i in range(threads):
                process_args.append(
                    [network_architecture, x_train, y_train, x_val, y_val, def_batch, max_epochs,
                     def_learnr, def_dropout, i])

            with Pool(threads) as p:
                losses = p.starmap(multi_search, process_args)

            # Score best
            score = min(losses)

            # Write results to file
            print(
                f"Architecture2D: {network_architecture}, Validation loss (Best of {threads}): {score}, Full losses: {losses}")
            with open("architecture_grid2D.txt", "a+") as f:
                f.write(
                    f"Architecture: {network_architecture}, Validation loss (Best of {threads}): {score}, Full losses: {losses}\n")

            if score < best_loss:
                best_loss = score
                best_architecture = network_architecture

    # Write results to file
    # print(f"Best architecture2D: {best_architecture}, Best validation loss: {best_loss}")
    # with open("best_params.txt", "a+") as f:
    #     f.write(f"Best architecture2D: {best_architecture}, Validation loss: {best_loss} \n")

    return best_architecture


def RegressorHyperParameterSearch1D(x, y, threads=8, max_epochs=2000):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.
    Arguments:
        Add whatever inputs you need.
    Returns:
        The function should return your optimised hyper-parameters.
    """
    # Parameter ranges for architectural optimization
    layer1_values = range(4, 80, 4)

    # Default optimizer parameters
    def_learnr = 0.0003
    def_batch = 64
    def_dropout = 0.2

    # Dataset setup
    x_train, y_train, x_val, y_val, x_test, y_test = partition_inputs(x, y)

    # Clear files
    # with open("architecture_grid2D.txt", "w+") as f:
    #     f.write(f"")

    # Search vars
    best_architecture = [13, 10, 10, 1]
    best_loss = 10 ** 10

    # Grid search over layer sizes
    for layer1 in layer1_values:
        # Construct network object
        network_architecture = [13, layer1, 1]

        # Multiprocessing section
        process_args = []

        for i in range(threads):
            process_args.append(
                [network_architecture, x_train, y_train, x_val, y_val, def_batch, max_epochs,
                 def_learnr, def_dropout, i])

        with Pool(threads) as p:
            losses = p.starmap(multi_search, process_args)

        # Score best
        score = min(losses)

        # Write results to file
        print(
            f"Architecture1D: {network_architecture}, Validation loss (Best of {threads}): {score}, Full losses: {losses}")
        with open("architecture_grid1D.txt", "a+") as f:
            f.write(
                f"Architecture1d: {network_architecture}, Validation loss (Best of {threads}): {score}, Full losses: {losses}\n")

        if score < best_loss:
            best_loss = score
            best_architecture = network_architecture

    # Write results to file
    # print(f"Best architecture2D: {best_architecture}, Best validation loss: {best_loss}")
    # with open("best_params.txt", "a+") as f:
    #     f.write(f"Best architecture2D: {best_architecture}, Validation loss: {best_loss} \n")

    return best_architecture

def RegressorHyperParameterSearchDropout(x, y, threads=8, max_epochs=2000):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.
    Arguments:
        Add whatever inputs you need.
    Returns:
        The function should return your optimised hyper-parameters.
    """
    # Default optimizer parameters
    def_learnr = 0.0003
    def_batch = 64

    # Dataset setup
    x_train, y_train, x_val, y_val, x_test, y_test = partition_inputs(x, y)

    # Search over dropout values
    dropout_values = np.linspace(0, 0.8, num=24).tolist()
    best_loss = 10 ** 10
    best_dropout = 0
    best_architecture = [13, 24, 36, 1]

    for dropout in dropout_values:
        # Construct network object
        network_architecture = best_architecture

        # Multiprocessing section
        process_args = []

        for i in range(threads):
            process_args.append(
                [best_architecture, x_train, y_train, x_val, y_val, def_batch, max_epochs,
                 def_learnr, dropout, i])

        with Pool(threads) as p:
            losses = p.starmap(multi_search, process_args)

        # Score best
        score = min(losses)

        # Write results to file
        print(f"Dropout: {dropout}, Validation loss (Best of {threads}): {score}, Full losses: {losses}")
        with open("dropout_search.txt", "a+") as f:
            f.write(f"Dropout: {dropout}, Validation loss (Best of {threads}): {score}, Full losses: {losses} \n")

        if score < best_loss:
            best_loss = score
            best_dropout = dropout

    # Write results to file
    print(f"Best dropout: {best_dropout}, Best validation loss: {best_loss}")
    with open("best_params.txt", "a+") as f:
        f.write(f"Best dropout: {best_dropout}, Validation loss: {best_loss} \n")

def RegressorHyperParameterSearchLearning(x, y, threads=8, max_epochs=2000):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.
    Arguments:
        Add whatever inputs you need.
    Returns:
        The function should return your optimised hyper-parameters.
    """
    # Default optimizer parameters
    def_dropout = 0.0695
    def_batch = 64

    # Dataset setup
    x_train, y_train, x_val, y_val, x_test, y_test = partition_inputs(x, y)

    # Search over dropout values
    learnr_values = np.geomspace(0.001, 0.00001, num=16).tolist()
    best_architecture = [13, 24, 36, 1]

    for learnr in learnr_values:
        # Construct network object

        # Multiprocessing section
        process_args = []

        for i in range(threads):
            process_args.append(
                [best_architecture, x_train, y_train, x_val, y_val, def_batch, max_epochs,
                 learnr, def_dropout, i])

        with Pool(threads) as p:
            losses = p.starmap(multi_search, process_args)

        # Score best
        score = min(losses)

        # Write results to file
        print(f"Learnrate: {learnr}, Validation loss (Best of {threads}): {score}, Full losses: {losses}")
        with open("learnr_search.txt", "a+") as f:
            f.write(
                f"Learnrate: {learnr}, Validation loss (Best of {threads}): {score}, Full losses: {losses} \n")

def RegressorHyperParameterSearchBatch(x, y, threads=8, max_epochs=2000):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.
    Arguments:
        Add whatever inputs you need.
    Returns:
        The function should return your optimised hyper-parameters.
    """
    # Default optimizer parameters
    def_dropout = 0.0695
    def_learnr = 0.000398

    # Dataset setup
    x_train, y_train, x_val, y_val, x_test, y_test = partition_inputs(x, y)

    # Search over dropout values
    batch_values = np.geomspace(2, 1024, num=10).tolist()
    # batch_values = [2, 4, 8]
    best_architecture = [13, 24, 36, 1]

    for batch in batch_values:
        # Construct network object
        # Multiprocessing section
        process_args = []

        for i in range(threads):
            process_args.append(
                [best_architecture, x_train, y_train, x_val, y_val, round(batch), max_epochs,
                 def_learnr, def_dropout, i])

        with Pool(threads) as p:
            losses = p.starmap(multi_search, process_args)

        # Score best
        score = min(losses)

        # Write results to file
        print(f"Batch: {batch}, Validation loss (Best of {threads}): {score}, Full losses: {losses}")
        with open("batch_search.txt", "a+") as f:
            f.write(f"batch: {batch}, Validation loss (Best of {threads}): {score}, Full losses: {losses} \n")