import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import re
import scipy.signal

def as_data_frame(df, copy=False, decode=False, skip_scalars=False):
    """
    Convert df to a dataframe if it is not already a dataframe.

    Args:
        df: dataframe or dict/NpzFile convertible to a dataframe.
        copy: if True, return a copy of df if it is already a dataframe.
        decode: if True, decode bytes to strings.
        skip_scalars: if True, skip columns with scalar values.
    """
    if isinstance(df, pd.DataFrame):
        if copy:
            df = df.copy()
        if decode:
            for key, value in df.items():
                if (value.dtype.kind == 'S'
                        or (value.dtype == np.dtype('O')
                            and isinstance(value[0], bytes))):
                    df[key] = [x.decode("utf-8") for x in value]
        return df
    df_type = type(df)
    type_check = isinstance(df, dict) or isinstance(df, np.lib.npyio.NpzFile)
    try:
        if not isinstance(df, dict):
            df = dict(df.items())
        elif decode or skip_scalars:
            df = df.copy()
        if decode:
            # df is a dict and copied already.
            for key, value in df.items():
                if (value.dtype.kind == 'S'
                        or (value.dtype == np.dtype('O')
                            and isinstance(value[0], bytes))):
                    df[key] = [x.decode("utf-8") for x in value]
        if skip_scalars:
            # df is a dict and copied already.
            for key in list(df.keys()):
                value = df[key]
                if (isinstance(value, np.ndarray) and value.shape == ()) or (
                        isinstance(value, str) or isinstance(value, bytes)
                        or isinstance(value, float) or isinstance(value, int)):
                    del df[key]
        return pd.DataFrame(df)
    except (TypeError, AttributeError, ValueError) as e:
        if type_check:
            raise
        raise ValueError(
            "df must be a dataframe "
            + "or a dict/NpzFile convertible to a dataframe, "
            + f"not {df_type}."
        ) from e

# The following color map was inspired by the playground
# https://playground.tensorflow.org/
# which includes a spiral dataset.
# If this color map is used for values between -1 and 1,
# then, similarly to the original color map, 0 is white.
# However, the differences include:
# * colors are more saturated here,
# * negative values correspond to blue, which is more intuitive since
#   blue is usually associated to cold, i.e. negative (in Celsius) temperatures.
# * positive values correspond to red, which is more intuitive since
#   red is usually associated to hot, i.e. positive temperatures.
spiral_background2_cmap_colors = [
        (0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 0), (1, 0, 0),
        (1, 0, 1)]

spiral_background2_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'spiral_background2', spiral_background2_cmap_colors, N=256)

def mesh_convert(x, y, z):
    """
    Converts 3 1D arrays into a meshgrid (xx, yy, zz).

    Args:
        x,y,z: 1D arrays of the same length nn.
        Caller guarantees that nn = nx * ny where
        nx is the number of unique elements in x
        and ny is the number of unique elements in y.
        Moreover, any pair (x[i], y[i]) is unique.

    Returns: tuple (xx, yy, zz), where
        xx = np.unique(x)
        yy = np.unique(y)
        zz is a 2D array of shape (nx, ny) such that
        zz[i, j] = z[k] where (x[k], y[k]) = (xx[i], yy[j])
    """
    nn = x.shape[0]
    assert x.shape == (nn,)
    assert y.shape == (nn,)
    assert z.shape == (nn,), f"{z.shape =} != ({nn},)"
    xx = np.unique(x)
    yy = np.unique(y)
    assert xx.shape[0] * yy.shape[0] == nn, f"{xx.shape[0]} * {yy.shape[0]} != {nn}"
    ii = np.lexsort((y, x))
    zz = z[ii].reshape(xx.shape[0], yy.shape[0])
    return xx, yy, zz

def df_to_meshgrid(df, x_col='x', y_col='y'):
    """
    Converts a dataframe to a dictionary with meshgrid-like data.

    The output format is compatible with matplotlib's pcolormesh:
    ```
    mesh_dict = df_to_meshgrid(df)
    plt.pcolormesh(mesh_dict['x'], mesh_dict['y'], mesh_dict['z'])
    ```

    Args:
        df: dataframe with columns "x", "y", and other columns.
            Caller should guarantee that df has exactly one row
            for each pair (x, y) where x in df["x"] and y in df["y"].
            If df is not a dataframe, an conversion attempt is made.
        x_col: name of the column with x values to use instead of "x".
        y_col: name of the column with y values to use instead of "y".

    Returns:
        A dictionary with the same keys as the column names of df.
        Values corresponding to keys x_col and y_col
        are 1D np.ndarrays with unique values from the corresponding
        columns of df.
        Values corresponding to other keys are 2D np.ndarrays
        with shape (len(res[y_col]), len(res[x_col])).

    """
    df = as_data_frame(df, copy=True)
    x_values = np.unique(df[x_col])
    y_values = np.unique(df[y_col])
    x_len = len(x_values)
    y_len = len(y_values)

    x_to_idx = {x: i for i, x in enumerate(x_values)}
    y_to_idx = {y: i for i, y in enumerate(y_values)}

    df['x_idx'] = df[x_col].map(x_to_idx)
    df['y_idx'] = df[y_col].map(y_to_idx)

    res = {x_col: x_values, y_col: y_values}

    other_columns = set(df.columns) - {x_col, y_col, 'x_idx', 'y_idx'}
    for col in other_columns:
        res[col] = np.empty((y_len, x_len), dtype=df[col].dtype)
        res[col][df['y_idx'], df['x_idx']] = df[col].to_numpy()

    return res

def plot_fim_mgrid(
        ax, fim_mgrid, zz_max=None,
        lambda_max=63/64, xlim='auto', ylim='auto',
        xlabel="$\lambda_0$", ylabel="$\lambda_1$"):
    """
    Plots a meshgrid of fidelity susceptibility values.

    Args:
        ax: matplotlib axis.
        fim_mgrid: meshgrid of fidelity susceptibility values.
        zz_max: cutoff to scale the values plotted.
            If None, scale to the maximum value.
        lambda_max: used when xlim or ylim is 'auto'.
        xlim, ylim: limits for the x and y axes.
            If 'auto' use (0, lambda_max).
        xlabel, ylabel: labels for the x and y axes.

    Returns:
        zz_max: maximum value used for scaling.
    """
    label_size=18
    ax.tick_params(axis='both', which='major', labelsize=16)
    if xlim == 'auto':
        xlim = (0, lambda_max)
    if ylim == 'auto':
        ylim = (0, lambda_max)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)

    convert_df_chifc_to_fim(
            fim_mgrid, ensure_columns=["fim_00", "fim_01", "fim_11"])
    fim_a = fim_mgrid["fim_00"] / 4 + 3 * fim_mgrid["fim_11"] / 4
    fim_b = - (3**0.5 / 2) * fim_mgrid["fim_01"]
    zzs = np.array([
        fim_mgrid["fim_00"],
        fim_a + fim_b,
        fim_a - fim_b])
    zzs = np.maximum(zzs, 0)**0.5
    if zz_max is None:
        zz_max = np.max(zzs)
    zzs = zzs.transpose([1, 2, 0]) / zz_max
    zzs = np.minimum(1, zzs)
    zzs = np.maximum(0, zzs)
    ax.pcolormesh(
        fim_mgrid["lambda0"], fim_mgrid["lambda1"], 1 - zzs, shading='nearest')
    return zz_max

def plot_fim_mgrid_legend(
        ax, r2=1, r1=0.9, r0=0.8, x_resolution=512, y_resolution=512,
        xlim=(-1, 1), ylim=(-1, 1)):
    delta_x = xlim[1] - xlim[0]
    x_resolution = int(0.5 + x_resolution * abs(delta_x))
    delta_y = ylim[1] - ylim[0]
    y_resolution = int(0.5 + y_resolution * abs(delta_y))
    xx = np.linspace(*xlim, x_resolution)
    yy = np.linspace(*ylim, y_resolution)
    zz = np.zeros((y_resolution, x_resolution, 3))
    r = (xx[np.newaxis, :]**2 + yy[:, np.newaxis]**2)**0.5
    circle_ii = (r1 <= r) & (r < r2)
    v_vec = np.array(np.meshgrid(xx, yy)).transpose([1, 2, 0])
    v_dir = np.array([xx[np.newaxis, :]  / r, yy[:, np.newaxis] / r]).transpose([1, 2, 0])
    assert v_dir.shape == v_vec.shape
    color_dirs = np.array([[1, 0], [-1/2, 3**0.5 / 2], [-1/2, -3**0.5 / 2]])
    for i, color_dir in enumerate(color_dirs):
        zz[circle_ii, i] = np.abs(v_dir @ color_dir)[circle_ii]

    ball_ii = (r <= r0)
    for i, color_dir in enumerate(color_dirs):
        zz[ball_ii, i] = np.abs(v_vec @ color_dir)[ball_ii]

    ax.imshow(1 - zz, extent=[*xlim, *ylim], origin='lower')
    ax.set_axis_off()

def _plot_fim_df_1d_ml(ax, fixed_lambda_index, fixed_lambda_val, ml_fim_df,
                       lookup_tolerance=0, **kwargs):
    x_lambda_index = 1 - fixed_lambda_index
    fixed_lambda_colname = "lambda" + str(fixed_lambda_index)
    x_lambda_colname = "lambda" + str(x_lambda_index)
    ii1 = (np.abs(ml_fim_df[fixed_lambda_colname] - fixed_lambda_val)
           <= lookup_tolerance)
    if ii1.sum() == 0:
        raise ValueError(
            f"No rows matching ml_fim_df[{fixed_lambda_colname}] "
            + f"== {fixed_lambda_val}. Unique values are:\n"
            + f"{ml_fim_df[fixed_lambda_colname].unique()}")
    if lookup_tolerance > 0:
        unique_lambda_vals = ml_fim_df[fixed_lambda_colname][ii1].unique()
        if len(unique_lambda_vals) > 1:
            raise ValueError(
                f"Multiple rows matching ml_fim_df[{fixed_lambda_colname}] "
                + f"== {fixed_lambda_val} within tolerance "
                + f"{lookup_tolerance}. Unique values are:\n"
                + f"{unique_lambda_vals}")
    fim_dir_colname = "fim_" + str(x_lambda_index) * 2
    lambda1s, f11s = (
            ml_fim_df[key][ii1]
            for key in (x_lambda_colname, fim_dir_colname))
    default_style = dict(linestyle='-', color='blue', marker='.',
                         linewidth=1.0, markersize=2.0)
    ax.plot(lambda1s, f11s, **{**default_style, **kwargs})
    return lambda1s, f11s

def convert_df_chifc_to_fim(df, ensure_columns=None):
    """
    Adds 'fim*' columns when the corresponding 'chi*' columns are present.

    Args:
        df: A pandas DataFrame or dict to modify in-place.
        ensure_columns: If not None, a list of column names to ensure exist
            in the DataFrame. If any of these columns are missing, this
            will raise a ValueError.
    """
    if isinstance(df, dict):
        columns = list(df.keys())
    else:
        columns = df.columns.tolist()
    new_columns = []
    for col in columns:
        match = re.match(r'^chi(_fc|fc_\d{2})$', col)
        if not match:
            continue
        new_col = col.replace('chi_fc', 'fim').replace('chifc', 'fim')
        if new_col not in columns:
            df[new_col] = df[col] * 4
            new_columns.append(new_col)
    if ensure_columns is not None:
        columns += new_columns
        for col in ensure_columns:
            if col not in columns:
                raise ValueError(f"Missing column {col} (not in {columns})")

def grid_point_to_human_str(
        v, grid, resolution=None, lookup_tolerance=2**(-23)):
    """
    Converts a grid point (e.g. 27 / 64) to a human-readable string
    (e.g. "27 / 64").
    """
    if resolution is None:
        resolution = int(0.5 + (len(grid) - 1) / (grid[-1] - grid[0]))
    # Below we use 'num' for 'numerator' (denominator is `resolution`).
    num_start = int(0.5 + grid[0] * resolution)
    expected_grid = (num_start + np.arange(len(grid), dtype=grid.dtype)
            ) / resolution
    use_float = False
    if not np.allclose(grid, expected_grid, atol=lookup_tolerance):
        use_float = True
    num_v = int(0.5 + v * resolution)
    if abs(v - num_v / resolution) > lookup_tolerance:
        use_float = True
    if use_float:
        return f"{v:.3f}"
    return f"{num_v} / {resolution}"

def plot_fim_df_1d(
        fim_df=None,
        ml_fim_dfs=None,
        sm_name=None,
        ax=None,
        fixed_lambda=(0, 25),
        resolution=None,
        lookup_tolerance=2**(-23),
        file_name=None,
        fim_vlines=None,
        ymax=150,
        figsize=(12, 5),
        verbose=2,
        gt_label="Ground truth",
        ml_kwargs=None,
        savefig_kwargs=None):
    """
    Plot the fidelity susceptibility along a line in parameter space.

    Assumes the space of lambdas is 2D and one of the lambdas is fixed.

    Args:
        fim_df: DataFrame describing the reference FIM.
        ml_fim_dfs: list of DataFrames describing the FIMs for the ML models.
        sm_name: name of the statistical manifold.
        fim_vlines: one of the following:
            - None or False: do not plot vertical lines.
            - True: extract peaks from fim_df and plot vertical lines.
            - a list of sweep_lambda values to plot vertical lines at.
        ax: matplotlib axis to plot on (if None, a new figure is created).
        ml_kwargs: keyword arguments for the plot of the ML models.
    """

    res = {}
    fixed_lambda_index, fixed_lambda_int_val = fixed_lambda
    fixed_lambda_colname = "lambda" + str(fixed_lambda_index)
    x_lambda_index = 1 - fixed_lambda_index
    x_lambda_colname = "lambda" + str(x_lambda_index)
    if fim_df is not None:
        ii = (fim_df["dir"] == str(x_lambda_index))
        fixed_lambda_vals = np.unique(fim_df[fixed_lambda_colname][ii])
        x_lambda_vals = np.unique(fim_df[x_lambda_colname][ii])
    else:
        df = ml_fim_dfs[0]
        fixed_lambda_vals = np.unique(df[fixed_lambda_colname])
        x_lambda_vals = np.unique(df[x_lambda_colname])

    if isinstance(fixed_lambda_int_val, int):
        fixed_lambda_val = fixed_lambda_vals[fixed_lambda_int_val]
    else:
        assert isinstance(fixed_lambda_int_val, float)
        fixed_lambda_val = fixed_lambda_int_val
        fixed_lambda_int_val = np.searchsorted(
            fixed_lambda_vals, fixed_lambda_val)
        v1 = fixed_lambda_vals[fixed_lambda_int_val]
        if fixed_lambda_int_val > 0:
            v0 = fixed_lambda_vals[fixed_lambda_int_val - 1]
            if fixed_lambda_val - v0 < v1 - fixed_lambda_val:
                fixed_lambda_int_val -= 1
                v1 = v0
        assert np.abs(fixed_lambda_val - v1) < lookup_tolerance, (
                f"{fixed_lambda_val} != {v1} "
                + f"== {fixed_lambda_vals[fixed_lambda_int_val]}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sm_str = f" for {sm_name}" if sm_name else ""
    fixed_lambda_str = grid_point_to_human_str(
        fixed_lambda_val, fixed_lambda_vals, resolution=resolution,
        lookup_tolerance=lookup_tolerance)
    ax.set_title("FIM comparison"
                 + f"{sm_str} at $\\lambda_{fixed_lambda_index} "
                 + f"= {fixed_lambda_str}$")
    ax.set_xlabel("$\lambda_" + f"{x_lambda_index}$")
    ax.set_ylabel("$g_{" + str(x_lambda_index) * 2 + "}$")

    ax.set_xlim((1.5 * x_lambda_vals[0] - 0.5 * x_lambda_vals[1],
                 1.5 * x_lambda_vals[-1] - 0.5 * x_lambda_vals[-2]))
    # Note: if ymax is None, this only sets the lower limit:
    ax.set_ylim((0, ymax))
    if fim_df is not None:
        ii0 = np.arange(fim_df.shape[0])[
                fim_df[fixed_lambda_colname] == fixed_lambda_val]
        x0_lambdas = fim_df[x_lambda_colname].iloc[ii0]
        convert_df_chifc_to_fim(fim_df, ensure_columns=["fim"])
        fim0 = fim_df["fim"].iloc[ii0]
        ax.plot(x0_lambdas, fim0,
                linewidth=1.0, color='black', linestyle='--',
                label=gt_label, marker='.', markersize=1.0)
    if fim_vlines:
        if isinstance(fim_vlines, bool):
            if fim_df is None:
                raise ValueError("fim_df must be provided to plot vertical lines.")
            fim_slice = fim_df.iloc[ii0]
            vline_ids = scipy.signal.find_peaks(fim_slice["fim"], prominence=20)[0]
            fim_vlines = fim_slice[x_lambda_colname].iloc[vline_ids]
        fim_vlines = np.asarray(fim_vlines)
        res["fim_vlines"] = fim_vlines
        for vline in fim_vlines:
            ax.axvline(vline, color='black', linestyle=':', linewidth=1.0)
    ml_ys = []
    ml_fim_dfs = ml_fim_dfs or []
    if isinstance(ml_kwargs, list):
        assert len(ml_fim_dfs) == len(ml_kwargs), (
            f"{len(ml_fim_dfs)} != {len(ml_kwargs)}")
    for i, ml_fim_df in enumerate(ml_fim_dfs):
        convert_df_chifc_to_fim(ml_fim_df, ensure_columns=["fim_00"])
        if isinstance(ml_kwargs, list):
            cur_ml_kwargs = ml_kwargs[i].copy()
        elif isinstance(ml_kwargs, dict):
            cur_ml_kwargs = ml_kwargs.copy()
        elif ml_kwargs is None:
            cur_ml_kwargs = {}
        else:
            raise ValueError(
                f"Invalid ml_kwargs: {ml_kwargs} (type {type(ml_kwargs)})")
        if i == 0:
            cur_ml_kwargs["label"] = cur_ml_kwargs.get("label", "ClassiFIM")
        elif isinstance(ml_kwargs, dict):
            try:
                del cur_ml_kwargs["label"]
            except KeyError:
                pass
        xs, ys = _plot_fim_df_1d_ml(
                ax, fixed_lambda_index, fixed_lambda_val, ml_fim_df,
                lookup_tolerance=lookup_tolerance, **cur_ml_kwargs)
        ml_ys.append(ys)
        if i == 0:
            xs0 = xs
        else:
            assert np.all(xs == xs0)
    if len(ml_fim_dfs) > 1:
        # Shade ys range.
        ml_ys_mean = np.mean(ml_ys, axis=0)
        ml_ys_std = np.std(ml_ys, axis=0)
        ax.fill_between(xs0, ml_ys_mean - 2 * ml_ys_std, ml_ys_mean + 2 * ml_ys_std,
                        alpha=0.3, color='black')

    ax.legend()

    if verbose >= 2 and len(ml_fim_dfs) >= 2:
        print("Grayed out regions are 2 standard deviations from the mean.")

    if file_name is not None:
        file_name = file_name.format(
            fixed_lambda_index=fixed_lambda_index,
            fixed_lambda_int_val=fixed_lambda_int_val)
        savefig_kwargs = savefig_kwargs or {}
        fig.savefig(file_name, bbox_inches='tight', **savefig_kwargs)
        res["file_name"] = file_name
        if verbose >= 1:
            print(f"Saved to '{file_name}'")
    return res

def plot_w2d(
        ax, npz, sweep_axis=0, colorbar=None, cbar_ax=None,
        vmin=0.75, vmax=1.0, acc_name="accuracy"):
    """
    Plot a 2D phase diagram of the W method.

    Args:
        ax: matplotlib axis to plot on.
        npz: dict or npz file with keys 'lambda_sweep_thresholds',
            'lambda_fixed', 'accuracy'. 'lambda_sweep' is also accepted.
        sweep_axis: 0 or 1, axis along which the sweep is performed.
        colorbar: whether to plot a colorbar.
        cbar_ax: axis for the colorbar.
        vmin, vmax: minimum and maximum values for the color scale.
        acc_name: alternative name for the accuracy column.
    """
    if colorbar is None:
        colorbar = cbar_ax is not None
    try:
        x = npz['lambda_sweep_thresholds']
    except KeyError:
        x = npz['lambda_sweep']
    y = npz['lambda_fixed']
    z = npz[acc_name]
    if sweep_axis != 0:
        assert sweep_axis == 1
        x, y, z = y, x, z.T
    pc = ax.pcolormesh(
        x, y, z,
        cmap=spiral_background2_cmap,
        shading='nearest', vmin=vmin, vmax=vmax)
    if colorbar:
        fig = ax.get_figure()
        if cbar_ax:
            fig.colorbar(pc, cax=cbar_ax)
        else:
            fig.colorbar(pc)
    return pc

def plot_w1d(ax, npz, lambda_v, **kwargs):
    try:
        x = npz['lambda_sweep_thresholds']
    except KeyError:
        x = npz['lambda_sweep']
    i = np.searchsorted(npz['lambda_fixed'], lambda_v)
    if i + 1 < len(npz['lambda_fixed']):
        v, vnext = npz['lambda_fixed'][i:i+2]
        if abs(v - lambda_v) > abs(vnext - lambda_v):
            i += 1
    acc = npz['accuracy'][i]
    acc_min = np.min(acc)
    ax.plot(x, acc, **kwargs)
    ax.set_xlim((0, 1))
    ax.set_ylim((max(0.75, acc_min - 0.01), 1))

def ax_sweepline(
        ax, lambda_fixed, sweep_axis=0, color='black', linewidth=0.5, **kwargs):
    assert sweep_axis in [0, 1]
    f = ax.axhline if sweep_axis == 0 else ax.axvline
    f(lambda_fixed, color=color, linewidth=linewidth, **kwargs)

def plot_pca(ax, pca):
    n0, n1, num_components = pca.shape
    assert n0 == 64
    assert n1 == 64
    assert num_components >= 3
    lambda0s = np.arange(64) / 63
    lambda1s = np.arange(64) / 63
    pca3 = pca[:, :, :3]
    z = pca3 - np.min(pca3, axis=(0, 1))
    z = z / np.max(z, axis=(0, 1))
    ax.pcolormesh(lambda0s, lambda1s, z.swapaxes(0, 1), shading="nearest")
    ax.set_xlabel(r"$\lambda_0$")
    ax.set_ylabel(r"$\lambda_1$")
    ax.set_title("Phase diagram from PCA")

def _get_kwargs(*args):
    kwargs = {}
    for arg in args:
        if arg is not None:
            kwargs.update(**arg)
    return kwargs

def plot_gt_peaks_2d(
        ax, sweep_lambda_index, gt_peaks, style=None,
        style_inner=None, style_outer=None):
    assert sweep_lambda_index in [0, 1]
    xs = gt_peaks["lambda_sweep"]
    ys = gt_peaks["lambda_fixed"]
    if sweep_lambda_index == 1:
        xs, ys = ys, xs
    ii = (gt_peaks["is_inner"] == True)
    ax.plot(
        xs[ii], ys[ii], **_get_kwargs(
            dict(
                linestyle='None', marker='o', mfc='none', mec='black',
                label="GT inner"),
            style, style_inner))
    ax.plot(
        xs[~ii], ys[~ii], **_get_kwargs(
            dict(
                linestyle='None', marker='s', mfc='none', mec='green',
                label="GT outer"),
            style, style_outer))

def plot_predicted_peaks_2d(ax, sweep_lambda_index, predicted_peaks, **kwargs):
    assert sweep_lambda_index in [0, 1]
    xs = predicted_peaks["lambda_sweep"]
    ys = predicted_peaks["lambda_fixed"]
    if sweep_lambda_index == 1:
        xs, ys = ys, xs
    plot_kwargs = {
        'linestyle': 'None', 'marker': 'v', 'mfc': (1, 1, 1, 0.5),
        'mec': (0.3, 0.3, 0.3), 'label': "Predicted"}
    plot_kwargs.update(kwargs)
    ax.plot(xs, ys, **plot_kwargs)

def max_of_second_highest_slow(z: np.ndarray) -> float:
    # Check if the input is a 2D array
    if z.ndim != 2:
        raise ValueError("Input must be a 2D array")

    second_highest_values = []
    for row in z:
        unique_row = np.unique(row)  # Get unique elements in the row
        if len(unique_row) < 2:
            raise ValueError("Each row must have at least two unique elements")
        sorted_row = np.sort(unique_row)  # Sort the row
        second_highest = sorted_row[-2]  # Get the second highest value
        second_highest_values.append(second_highest)
    return max(second_highest_values)  # Return the maximum of the second highest values

def max_of_second_highest_fast(z: np.ndarray) -> float:
    assert z.ndim == 2, "Input must be a 2D array"
    return np.max(np.partition(z, -2, axis=1)[:, -2])
