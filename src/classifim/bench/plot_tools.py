import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import re

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
            return df.copy()
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
        ax, r2=1, r1=0.9, r0=0.8, x_resolution=1024, y_resolution=1025):
    xx = np.linspace(-1, 1, x_resolution)
    yy = np.linspace(-1, 1, y_resolution)
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

    # imshow's y axis goes from top to bottom,
    # the opposite of ours, hence we need to reverse it.
    ax.imshow(1 - zz[::-1, :, :])
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
    default_style = dict(linestyle='-', color='black', marker='.',
                         linewidth=0.5, markersize=2.0)
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

def plot_fim_df_1d(
        fim_df=None,
        ml_fim_dfs=None,
        hamiltonian_name=None,
        fixed_lambda=(0, 25),
        resolution=64,
        lookup_tolerance=2**(-23),
        file_name=None,
        ymax=150,
        figsize=(12, 5),
        verbose=2,
        gt_label="Ground truth",
        savefig_kwargs=None):
    """
    Plot the fidelity susceptibility along a line in parameter space.

    Assumes the space of lambdas is 2D and one of the lambdas is fixed.

    Args:
        fim_df: DataFrame describing the reference FIM.
        ml_fim_dfs: list of DataFrames describing the FIMs for the ML models.
    """

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

    fig, ax = plt.subplots(figsize=figsize)

    hamiltonian_str = f" for {hamiltonian_name}" if hamiltonian_name else ""
    assert (np.abs(fixed_lambda_val - fixed_lambda_int_val / resolution)
            < lookup_tolerance)
    ax.set_title("FIM comparison"
                 + f"{hamiltonian_str} at $\\lambda_{fixed_lambda_index} "
                 + f"= {int(fixed_lambda_int_val)} / {resolution}$")
    ax.set_xlabel("$\lambda_" + f"{x_lambda_index}$")
    ax.set_ylabel("$g_{" + str(x_lambda_index) * 2 + "}$")

    ax.set_xlim((1.5 * x_lambda_vals[0] - 0.5 * x_lambda_vals[1],
                 1.5 * x_lambda_vals[-1] - 0.5 * x_lambda_vals[-2]))
    if ymax is not None:
        ax.set_ylim((0, ymax))
    if fim_df is not None:
        ii0 = np.arange(fim_df.shape[0])[
                fim_df[fixed_lambda_colname] == fixed_lambda_val]
        x0_lambdas = fim_df[x_lambda_colname].iloc[ii0]
        convert_df_chifc_to_fim(fim_df, ensure_columns=["fim"])
        fim0 = fim_df["fim"].iloc[ii0]
        ax.plot(x0_lambdas, fim0,
                linewidth=2.0, color='blue', linestyle='-',
                label=gt_label, marker='.', markersize=1.0)
    ml_ys = []
    ml_fim_dfs = ml_fim_dfs or []
    for i, ml_fim_df in enumerate(ml_fim_dfs):
        convert_df_chifc_to_fim(ml_fim_df, ensure_columns=["fim_00"])
        ml_kwargs = {}
        if i == 0:
            ml_kwargs = {"label": "ClassiFIM"}
        xs, ys = _plot_fim_df_1d_ml(
                ax, fixed_lambda_index, fixed_lambda_val, ml_fim_df,
                lookup_tolerance=lookup_tolerance, **ml_kwargs)
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

    if verbose >= 2:
        print("Grayed out regions are 2 standard deviations from the mean.")

    if file_name is not None:
        file_name = file_name.format(
            fixed_lambda_index=fixed_lambda_index,
            fixed_lambda_int_val=fixed_lambda_int_val)
        savefig_kwargs = savefig_kwargs or {}
        fig.savefig(file_name, bbox_inches='tight', **savefig_kwargs)
        if verbose >= 1:
            print(f"Saved to '{file_name}'")

