"""
Pipeline for data loading, training, and evaluation.
"""

import classifim.data_loading
import classifim.io
import classifim.layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import classifim.utils

def unpack_zs(zs, n_sites):
    """
    Converts an array of (2 * self.n_sites)-bit integers into an array of Z
        values in {-1, 1}.

    Input format: 0bTT...TTBB...BB
    Bit Bj of zs is (zs & (1 << j)) > 0.
    Bit Tj of zs is (zs & (1 << (self.nsites + j))) > 0.

    Output format: [[Z_{B0}, Z_{B1}, ..., Z_{B11}], [Z_{T0}, ..., Z_{T11}]]
    Z_{Bj} of zs_out is zs_out[..., 0, j].
    Z_{Tj} of zs_out is zs_out[..., 1, j].
    """
    zs_bin = classifim.utils.unpackbits(zs, 2 * n_sites)
    zs_bin = (1 - 2 * zs_bin).reshape(*zs.shape, 2, n_sites)
    return zs_bin.astype(np.float32)

class Pipeline:
    """
    Example usage:
    ```
    config = {
        "dataset_filename": "data.npz",
        "seed": 0,
        "device": "cuda:0"}
    pipeline = Pipeline(config)
    pipeline.load_data()
    pipeline.init_model()
    pipeline.train()
    pipeline.save_model(filename="model.pth")
    pipeline.load_model(filename="model.pth")
    pipeline.test()
    pipeline.eval_chifc()
    pipeline.save_chifc(filename="chifc.npz")
    ```
    """
    DEFAULT_CONFIG = {
        "device": "cuda:0",
        "num_epochs": 319}

    def __init__(self, config):
        """
        Args:
            config: A dictionary with the following keys:
                dataset_filename
                seed
                [optional] any keys from DEFAULT_CONFIG
        """
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.n_sites = self.config.get("n_sites", 12)

    def unpack_zs(self, zs):
        return unpack_zs(zs, self.n_sites)

    def with_unpacked_zs(self, dataset):
        """
        Returns a copy of dataset with the zs column unpacked.
        """
        dataset = dataset.copy()
        dataset["zs"] = self.unpack_zs(dataset["zs"])
        return dataset

    def load_data(self):
        """
        Initializes:
            self.seed
            self.prng

            # lambda in [-1, 1), zs is packed:
            self.dataset_train
            self.dataset_test

            # 100 passes, lambda in [-1, 1), zs is unpacked:
            self.dataset_bschifc_test
        Returns: None
        """
        npz_dataset = np.load(self.config["dataset_filename"])
        self.seed = self.config.get("seed")
        if self.seed is None:
            self.seed = npz_dataset["seed"]
        self.prng = classifim.utils.DeterministicPrng(self.seed)
        if "n_sites" in npz_dataset:
            assert self.n_sites == npz_dataset["n_sites"]

        self.dataset_train, self.dataset_test = classifim.io.split_train_test(
            npz_dataset, test_size=self.config.get("test_fraction", 0.1),
            seed=self.prng.get_seed("test"),
            scalar_keys=self.config.get("scalar_keys", ["seed", "dataset_i"]))
        for dataset in (self.dataset_train, self.dataset_test):
            classifim.io.scale_lambdas(
                    dataset, inplace=True, dtype=np.float32)
        self.dataset_bschifc_test = classifim.io.get_classifim_train_dataset(
            self.with_unpacked_zs(self.dataset_test),
            num_passes=self.config.get("test_num_passes", 100),
            seed=self.prng.get_seed("bitchifc_test"))

    @staticmethod
    @torch.no_grad()
    def test_nn(dataloader, model, loss_fn, device="cuda:0"):
        model.eval()
        model.to(device)
        running_loss = 0.0
        num_points = 0
        for data in dataloader:
            torch_data = [d.to(device, non_blocking=True) for d in data]
            labels = torch_data[-1]
            torch_data = torch_data[:-1]
            _, output2 = model(*data_torch)
            loss = loss_fn(output2, labels)
            cur_num_points = lambda0.shape[0]
            num_points += cur_num_points
            running_loss += loss.item() * cur_num_points
        running_loss /= num_points
        print(f"Avg loss: {running_loss:>8f} ({num_points} points)\n")
        return {"loss": running_loss, "num_points": num_points}

    def construct_model(self, device=None):
        model_init_kwargs = self.config.get("model_init_kwargs", {})
        model = classifim.layers.TwelveSitesNN2(**model_init_kwargs)
        if device is not None:
            model = model.to(device)
        self.model = model

    def get_train_loader(self, device=None):
        if device is None:
            device = self.config["device"]
        return classifim.data_loading.TwelveSitesDataLoader5(
            self.dataset_train, batch_size=2**14, device=device,
            n_sites=self.n_sites)

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
        max_lr = 0.01
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=max_lr/2, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=max_lr, pct_start=0.1,
            epochs=self.config["num_epochs"], steps_per_epoch=1)

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

    @staticmethod
    def train_nn(
            train_loader, model, loss_fn, optimizer, scheduler,
            num_epochs, device="cuda:0", verbose=1):
        """
        Args:
            verbose: integer >= 0, larger means more verbose:
                0: no output
                1: short output (a few lines)
                2: one line per epoch
        """
        model.train()
        res = {"epoch": [], "loss": [], "learning_rate": []}

        if verbose == 1:
            print(f"{num_epochs=}: ", end="")
        total_num_points = 0
        for epoch in range(num_epochs):
            num_points = 0
            running_loss = 0.0
            for data in train_loader:
                torch_data = [d.to(device, non_blocking=True) for d in data]
                labels = torch_data[-1]
                torch_data = torch_data[:-1]

                # Forward pass
                _, output2 = model(*data_torch)
                loss = loss_fn(output2, labels)
                # Backprop:
                # set_to_none optimization described at
                # https://tigress-web.princeton.edu/~jdh4/PyTorchPerformanceTuningGuide_GTC2021.pdf
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Collect statistics
                cur_num_points = lambda0.shape[0]
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
            total_num_points += num_points

        if verbose >= 1:
            print(f"Training finished ({total_num_points=}).")
        res["total_num_points"] = total_num_points
        return res

    def train(self):
        """
        Trains self.model
        """
        torch.manual_seed(self.prng.get_int_seed("train"))
        self.train_res = self.train_nn(
            self.train_loader, self.model, self.loss_fn, self.optimizer,
            self.scheduler, num_epochs=self.config["num_epochs"],
            device=self.config["device"])
        return self.train_res

    def plot_train_res(self):
        """
        Plots the training results.
        """
        loss = np.array(self.train_res["loss"])
        fig, ax = plt.subplots()
        ax.set_xlim((0, len(loss)))
        ax.set_ylim((np.min(loss)-0.003, np.log(2)))
        ax.plot(loss)
        fig.show()

    def _get_model_filename(self, filename=None):
        if filename is None:
            filename = self.config["model_filename"]
        return filename.format(seed=self.seed)

    def save_model(self, filename=None, verbose=True):
        """
        Saves self.model to filename
        """
        filename = self._get_model_filename(filename)
        torch.save(self.model.state_dict(), filename)
        if verbose:
            print(f"Saved model to '{filename}'")

    def load_model(self, filename=None, verbose=True):
        """
        Loads self.model from filename
        """
        filename = self._get_model_filename(filename)
        self.construct_model()
        self.model.load_state_dict(torch.load(filename))
        if verbose:
            print(f"Loaded model from '{filename}'")

    def get_test_loader(self):
        """
        Sets the seed and returns a DataLoader for the test set.
        """
        torch.manual_seed(self.prng.get_int_seed("test_model"))
        device = self.config["device"]
        dataset_bschifc_test2 = classifim.data_loading.TwelveSitesDataLoader3.prepare_dataset(
                self.with_unpacked_zs(self.dataset_test),
                self.dataset_bschifc_test,
                n_sites=self.n_sites)
        test_loader2 = classifim.data_loading.TwelveSitesDataLoader3(
                dataset_bschifc_test2, batch_size=131072, device=device,
                n_sites=self.n_sites)
        return test_loader2

    def test(self, dataloader=None):
        """
        Tests self.model
        """
        device = self.config["device"]
        if dataloader is None:
            test_loader2 = self.get_test_loader()
        else:
            test_loader2 = dataloader
        # Note: do not set seed here:
        # 1. It is already set in get_test_loader.
        # 2. We already ran this, recorded the results, and want to keep
        # them reproducible.
        return self.test_nn(test_loader2, self.model, self.loss_fn, device=device)

    def _convert_test_data_to_bench_format(self, data, copy=True):
        """
        Converts the test data to the format used by classifim_bench.

        Input zs format: [[Z_{B0}, Z_{T0}], ..., [Z_{B11}, Z_{T11}]]
        Output zs format: 0bT[11]T[10]...T[0]B[11]...B[0]
        """
        if copy:
            data = data.copy()
        num_samples = data["lambda0s"].shape[0]
        data["lambda0s"] = (data["lambda0s"] + 1) / 2
        data["dlambdas"] = data["dlambdas"] / 2
        zs_01 = (1 - data["zs"]) / 2
        assert zs_01.shape == (num_samples, self.n_sites, 2)
        zs_01 = zs_01.transpose((0, 2, 1)).reshape(
                (num_samples, 2 * self.n_sites))
        data["zs"] = classifim.utils.packbits(zs_01.astype(np.int32))
        return data

    def record_test_data(self, use_bench_format=False, dataloader=None):
        """
        Records the test data to a dict.

        Args:
            use_bench_format: if True, converts the data to the format used
                by classifim_bench.
        """
        device = self.config["device"]
        if dataloader is None:
            dataloader = self.get_test_loader()
        dump = {key: [] for key in dataloader.keys}
        for data in dataloader:
            # lambda0, dlambda, zs, labels:
            cur_dump = [d.to(device, non_blocking=True) for d in data]
            for key, val in zip(dump.keys(), cur_dump):
                dump[key].append(val.cpu().numpy())
        dump = {key: np.concatenate(val) for key, val in dump.items()}
        if use_bench_format:
            dump = self._convert_test_data_to_bench_format(dump, copy=False)
        return dump

    @staticmethod
    @torch.no_grad()
    def eval_chifc_estimate(dataset, model, device="cuda:0", batch_size=4096):
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
        lambdas = dataset["lambdas"]
        zs = dataset["zs"]

        num_samples = lambdas.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        output1_all = np.zeros((num_samples, 2), dtype=np.float32)

        model.eval()
        model.to(device)
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_samples)

            batch_lambdas = torch.from_numpy(lambdas[batch_start:batch_end]).to(device)
            batch_zs = torch.from_numpy(zs[batch_start:batch_end]).transpose(1, 2).to(device)

            dlambdas = torch.zeros(batch_lambdas.shape, device=device)
            output1, _ = model(batch_lambdas, dlambdas, batch_zs)

            output1_all[batch_start:batch_end] = output1.cpu().numpy()

        # Scale lambdas from [-1, 1] back to [0, 1]
        return ((lambdas + 1) / 2, output1_all)

    def eval_chifc(self):
        lambdas, output1 = self.eval_chifc_estimate(
            self.with_unpacked_zs(self.dataset_train),
            self.model, batch_size=8192,
            device=self.config["device"])

        # Convert to float64 to avoid significantly negative
        # eigenvalues of chifc.
        output1 = output1.astype(np.float64)

        # Note that chifc thus obtained should be multiplied by:
        # * 4 (since NN uses 2 * (lambda - 0.5) instead of lambda)
        # * (1/4) (since we are estimating chifc, not FIM)
        # These two factors cancel out, so we don't need to multiply by anything.
        df = pd.DataFrame({
            "lambda0": lambdas[:, 0],
            "lambda1": lambdas[:, 1],
            "chifc_00": output1[:, 0] * output1[:, 0],
            "chifc_01": output1[:, 0] * output1[:, 1],
            "chifc_11": output1[:, 1] * output1[:, 1]})
        df_summary = df.groupby(["lambda0", "lambda1"]).agg(
            cnt=("lambda1", "count"),
            **{col: (col, 'mean') for col in df.columns if col not in ('lambda0', 'lambda1')})
        df_summary.reset_index(inplace=True)
        df_summary
        self.chi_fc = {key: df_summary[key].to_numpy() for key in df.columns}
        return self.chi_fc

    def _get_chifc_filename(self, filename=None):
        if filename is None:
            filename = self.config["chifc_filename"]
        return filename.format(seed=self.seed)

    def save_chifc(self, filename=None, verbose=True):
        """
        Saves self.chi_fc to filename
        """
        filename = self._get_chifc_filename(filename)
        assert hasattr(self, "chi_fc"), "self.chi_fc is not set"
        assert self.chi_fc is not None, "self.chi_fc is None"
        np.savez(filename, **self.chi_fc)
        if verbose:
            print(f"Saved chi_fc to '{filename}'")
