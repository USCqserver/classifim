import classifim.io
import classifim.utils
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

class Pipeline:
    """
    Pipeline for training and using ClassiFIM BC neural network.
    """
    def __init__(self, config, dataset=None, prng=None):
        """
        Initialize the pipeline.

        Args:
            config: Config object.
            dataset: dict with the dataset to be split into train and test.
            prng: Deterministic pseudo-random number generator.
        """
        self.config = config
        if dataset is None:
            dataset = np.load(self.config["dataset_filename"])
        self.dataset = dataset
        if prng is None:
            prng = classifim.utils.DeterministicPrng(self.config["suffix"])
        self.prng = prng
        self.split_dataset()

    def fit_transform_lambdas(self, dataset):
        """
        Scale the lambdas to prepare for ClassiFIM BC training.

        This function should save the scaling parameters for future use
        in self.transform_lambdas.
        Dataset is transformed in-place.
        """
        self.lambda_scaler = MinMaxScaler(feature_range=(-1, 1))
        dataset["scaled_lambdas"] = self.lambda_scaler.fit_transform(
            dataset["lambdas"])

    def transform_lambdas(self, dataset):
        """
        Scale the lambdas using the parameters saved in self.fit_transform_lambdas.
        Dataset is transformed in-place.
        """
        dataset["scaled_lambdas"] = self.lambda_scaler.transform(
            dataset["lambdas"])

    def fit_transform(self, dataset):
        """
        Scale the dataset to prepare for ClassiFIM BC training.

        This function should save the scaling parameters for future use
        in self.transform.
        Dataset is transformed in-place.
        """
        self.fit_transform_lambdas(dataset)

    def transform(self, dataset):
        """
        Scale the dataset using the parameters saved in self.fit_transform.
        Dataset is transformed in-place.
        """
        self.transform_lambdas(dataset)

    def split_dataset(self):
        """
        Split the dataset into train and test.
        """
        test_fraction = self.config.get("test_fraction", 0.1)
        dataset_train, dataset_test = classifim.io.split_train_test(
            self.dataset,
            test_size=test_fraction,
            seed=self.prng.get_seed("split_test"),
            scalar_keys=self.config["scalar_keys"])
        if self.config["hold_out_test"]:
            dataset_train, dataset_test = classifim.io.split_train_test(
                dataset_train,
                test_size=self.config.get("val_fraction", test_fraction),
                seed=self.prng.get_seed("split_val"),
                scalar_keys=self.config["scalar_keys"])
        self.fit_transform(dataset_train)
        self.transform(dataset_test)
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test

    def _get_data_loader(self, dataset, is_train, batch_size=None, device=None):
        raise NotImplementedError

    def get_train_loader(self, batch_size=None, device=None):
        """
        Construct a DataLoader for the train set.

        This can use `self.dataset_train` and `self.config`.

        Often the returned object is not torch.utils.data.DataLoader
        for efficiency reasons. The method should be implemented in
        derived classes. Alternatively, the method `_get_data_loader`
        should be implemented.
        """
        return self._get_data_loader(
            self.dataset_train, is_train=True, device=device)

    def get_test_loader(self, batch_size=None, device=None):
        """
        Construct a DataLoader for the test set.

        This can use `self.dataset_test` and `self.config`.

        Often the returned object is not torch.utils.data.DataLoader
        for efficiency reasons. The method should be implemented in
        derived classes.
        """
        return self._get_data_loader(
            self.dataset_train, batch_size=batch_size, is_train=False,
            device=device)

    def get_fim_loader(self, dataset, batch_size=None, device=None):
        if device is None:
            device = self.config["device"]
        if batch_size is None:
            batch_size = self.config.get("batch_size", 2**14)
        raise NotImplementedError()

    def init_model(self):
        """
        Initializes:
            self.model
            self.loss_fn
            self.optimizer
            self.scheduler
            self.train_loader
        Returns: None
        """
        torch.manual_seed(self.prng.get_int_seed("init_model"))
        device = self.config["device"]
        self.train_loader = self.get_train_loader(device=device)

        # Initialize the network, loss function, and optimizer
        self.construct_model(device=device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        max_lr = self.config.get("max_lr", 1e-2)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=max_lr/2,
            weight_decay=self.config.get("weight_decay", 1e-4))
        pct_start = self.config.get("one_cycle_lr.pct_start", 0.1)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=max_lr, pct_start=pct_start,
            epochs=self.config["num_epochs"], steps_per_epoch=1)

    def construct_model(self, device=None, cls=None):
        if cls is None:
            raise NotImplementedError
        model_init_kwargs = self.config.get("model_init_kwargs", {})
        model = cls(**model_init_kwargs)
        if device is not None:
            model = model.to(device)
        self.model = model


    def cleanup_after_training(self):
        """
        Deletes some data created by init_model:
            self.optimizer
            self.scheduler
            self.train_loader
        Keeps self.model, self.loss_fn.
        Returns: None
        """
        del self.optimizer
        del self.scheduler
        del self.train_loader

    def train(self):
        """
        Trains self.model
        """
        if not hasattr(self, "model") or self.model is None:
            self.init_model()
        torch.manual_seed(self.prng.get_int_seed("train"))
        self.train_res = self.train_nn(
            self.train_loader, self.model, self.loss_fn, self.optimizer,
            self.scheduler, num_epochs=self.config["num_epochs"],
            device=self.config["device"])
        return self.train_res

    @staticmethod
    def train_nn(
            train_loader, model, loss_fn, optimizer, scheduler,
            num_epochs, device="cuda:0", verbose=1):
        """
        Train ClassiFIM BC neural network.

        Args:
            verbose: integer >= 0, larger means more verbose:
                0: no output
                1: short output (a few lines)
                2: one line per epoch
        """
        model.train()
        res = {"epoch": [], "loss": [], "learning_rate": []}

        if verbose >= 1:
            print(f"{datetime.datetime.now()}: Training started")
        if verbose == 1:
            print(f"{num_epochs=}: ", end="")
        for epoch in range(num_epochs):
            num_points = 0
            running_loss = 0.0
            for data in train_loader:
                torch_data = [d.to(device, non_blocking=True) for d in data]
                labels = torch_data[-1]
                torch_data = torch_data[:-1]

                # Forward pass
                _, output2 = model(*torch_data)
                loss = loss_fn(output2, labels)
                # Backprop:
                # set_to_none optimization described at
                # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Collect statistics
                cur_num_points = torch_data[0].shape[0]
                running_loss += loss.item() * cur_num_points
                num_points += cur_num_points
            last_lr = scheduler.get_last_lr()[0]
            if verbose >= 2:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / num_points}, LR: {last_lr:.4g}")
            elif verbose == 1:
                print(f"{epoch + 1}:{running_loss / num_points:.4g}", end=" ")
            scheduler.step()
            train_loader.reshuffle()
            res["epoch"].append(epoch)
            res["loss"].append(running_loss / num_points)
            res["learning_rate"].append(last_lr)
            res["num_points"] = num_points
        if verbose == 1:
            print()
        return res

    def test(self, num_epochs=None, batch_size=2**14, device="cuda:0"):
        if num_epochs is None:
            num_epochs = self.config["num_epochs"]
        torch.manual_seed(self.prng.get_int_seed("test_model"))
        test_loader = self.get_test_loader()
        self.test_res = self.test_nn(
            test_loader, self.model, self.loss_fn,
            num_epochs=num_epochs,
            device=self.config["device"])
        return self.test_res

    @staticmethod
    @torch.no_grad()
    def test_nn(data_loader, model, loss_fn, num_epochs=1, device="cuda:0"):
        model.eval()
        model.to(device)
        running_loss = 0.0
        num_points = 0
        for epoch in range(num_epochs):
            for data in data_loader:
                torch_data = [d.to(device, non_blocking=True) for d in data]
                labels = torch_data[-1]
                torch_data = torch_data[:-1]
                _, output2 = model(*torch_data)
                loss = loss_fn(output2, labels)
                cur_num_points = torch_data[0].shape[0]
                num_points += cur_num_points
                running_loss += loss.item() * cur_num_points
            # Note: reshuffle changes the pairings (lambda_m, lambda_p) and,
            # therefore, the loss is not the same on every iteration:
            data_loader.reshuffle()

        running_loss /= num_points
        print(f"Avg loss: {running_loss:>8f} ({num_points} points)\n")
        return {"loss": running_loss, "num_points": num_points}

    def _get_fim_filename(self, filename=None):
        if filename is None:
            filename = self.config["fim_filename"]
        return filename.format(**self.config)

    def _get_model_filename(self, filename=None):
        if filename is None:
            filename = self.config["model_filename"]
        return filename.format(**self.config)

    def save_fim(self, filename=None, verbose=True):
        """
        Saves self.fim to filename
        """
        assert hasattr(self, "fim"), "self.fim is not set"
        assert self.fim is not None, "self.fim is None"
        filename = self._get_fim_filename(filename)
        np.savez(filename, **self.fim)
        if verbose:
            print(f"Saved FIM to '{filename}'")

    @torch.no_grad()
    def eval_fim_estimate(self, dataset, model, device="cuda:0", batch_size=None):
        """
        Args:
            dataset: dict with the following keys:
            - lambdas: numpy array of shape (num_samples, 2) on [-1, 1] scale.
            - zs: numpy array of shape (num_samples, 2, n_sites).

        Returns:
            lambdas: numpy array of shape (num_samples, 2).
                (same as dataset['lambdas'] but scaled to [0, 1) instead of [-1, 1)).
            output1: numpy array of shape (num_samples, 2). Output of the model
                used to estimate fidelity susceptibility.
        """
        lambdas_np = dataset["scaled_lambdas"].astype(np.float32)
        lambdas_torch = torch.from_numpy(lambdas_np).to(device)
        num_samples, num_lambdas = lambdas_np.shape
        output1_all = np.zeros((num_samples, num_lambdas), dtype=np.float32)
        fim_loader = self.get_fim_loader(
            dataset, batch_size=batch_size, device=device)

        model.eval()
        model.to(device)
        actual_num_samples = 0
        for lambda_range, z_data in fim_loader:
            cur_lambdas = lambdas_torch[lambda_range]
            actual_num_samples += cur_lambdas.shape[0]
            dlambdas = torch.zeros(cur_lambdas.shape, device=device)
            output1, _ = model(cur_lambdas, dlambdas, *z_data)
            output1_all[lambda_range] = output1.cpu().numpy()
        assert actual_num_samples == num_samples

        # Scale lambdas from [-1, 1] back to [0, 1]
        return ((lambdas_np + 1) / 2, output1_all)

    def eval_fim(self):
        lambdas, output1 = self.eval_fim_estimate(
            self.dataset_train,
            self.model, batch_size=8192,
            device=self.config["device"])

        # Convert to float64 to avoid significantly negative
        # eigenvalues of FIM.
        output1 = output1.astype(np.float64)

        df = dict()
        # Encoding index as a digit only works if index is < 10:
        assert lambdas.shape[1] <= 10

        lambda_names = []
        for i in range(lambdas.shape[1]):
            lambda_name = f"lambda{i}"
            df[lambda_name] = lambdas[:, i]
            lambda_names.append(lambda_name)

        for i in range(output1.shape[1]):
            for j in range(i, output1.shape[1]):
                df[f"fim_{i}{j}"] = 4 * output1[:, i] * output1[:, j]

        df = pd.DataFrame(df)
        non_lambda_columns = [
            col for col in df.columns if col not in lambda_names]
        df_summary = df.groupby(lambda_names).agg(
            cnt=("lambda0", "count"),
            **{col: (col, 'mean') for col in non_lambda_columns})
        df_summary.reset_index(inplace=True)
        self.fim = {key: df_summary[key].to_numpy() for key in df.columns}
        self.fim["description"] = (
            "FIM with respect to {lambdas scaled to [0, 1]}")

        return self.fim

    def save_model(self, filename=None, verbose=True):
        filename = self._get_model_filename(filename)
        torch.save(self.model.state_dict(), filename)
        if verbose:
            print(f"Saved model to '{filename}'")

    def load_model(self, filename=None, verbose=True):
        filename = self._get_model_filename(filename)
        self.construct_model()
        self.model.load_state_dict(torch.load(filename))
        if verbose:
            print(f"Loaded model from '{filename}'")

    @staticmethod
    def _divide_into_batches(data, batch_size):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def apply_model(self, data, batch_size=None, device=None):
        """
        Applies the model to the data.
        """
        if batch_size is None:
            batch_size = self.config.get("batch_size", 2**14)
        if device is None:
            device = self.config["device"]
        data = [self._divide_into_batches(d, batch_size) for d in data]
        self.model.eval()
        self.model.to(device)
        output_is_tuple = None
        with torch.no_grad():
            output = []
            for batch in zip(*data):
                cur_output = self.model(*[d.to(device) for d in batch])
                if output_is_tuple is None:
                    output_is_tuple = isinstance(cur_output, tuple)
                if not output_is_tuple:
                    cur_output = (cur_output,)
                cur_output = [d.cpu().numpy() for d in cur_output]
                output.append(cur_output)
            output = [np.concatenate(d) for d in zip(*output)]
            if output_is_tuple:
                return tuple(output)
            else:
                return output[0]

