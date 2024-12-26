import numpy as np
import xarray as xr
from tigramite import data_processing as pp
import matplotlib.pyplot as plt
import seaborn as sns

i, j, k, l = 3, 3, 3, 3
# i, j, k, l = 3, 2, 1, 2
# i, j, k, l = 3, 2, 3, 2
# i, j, k, l = 15, 6, 4, 15

xdata = xr.open_dataset('AirTempData.nc')

# ENSO LAT 6,-6, LON 190, 240
# BCT LAT 65,50 LON 200, 240
# TATL LAT 25, 5, LON 305, 325

# for i in range(2, 5):
#     for j in range(2, 5):
#         for k in range(1, 4):
#             for l in range(2, 5):
Xregion = xdata.sel(lat=slice(6., -6., k),
                    lon=slice(190., 240., i))
Yregion = xdata.sel(lat=slice(65., 50., j),
                    lon=slice(200., 240., l))

# de-seasonlize
# ----------------
monthlymean = Xregion.groupby("time.month").mean("time")

anomalies_Xregion = Xregion.groupby("time.month") - monthlymean

Yregion_monthlymean = Yregion.groupby(
    "time.month").mean("time")

anomalies_Yregion = Yregion.groupby(
    "time.month") - Yregion_monthlymean

# functions to consider triples on months
# -----------------------------------------


def is_ond(month):
    return (month >= 10) & (month <= 12)


def is_son(month):
    return (month >= 9) & (month <= 11)


def is_ndj(month):
    return ((month >= 11) & (month <= 12)) or (month == 1)


def is_jfm(month):
    return (month >= 1) & (month <= 3)

# NINO for oct-nov-dec
# --------------------


ond_Xregion = anomalies_Xregion.sel(
    time=is_ond(xdata['time.month']))

ond_Xregion_by_year = ond_Xregion.groupby("time.year").mean()

num_ond_Xregion = np.array(ond_Xregion_by_year.to_array())[0]

reshaped_Xregion = np.reshape(num_ond_Xregion, newshape=(
    num_ond_Xregion.shape[0], num_ond_Xregion.shape[1]*num_ond_Xregion.shape[2]))

# BCT for jan-feb-mar
# -------------------

jfm_Yregion = anomalies_Yregion.sel(
    time=is_jfm(xdata['time.month']))

jfm_Yregion_by_year = jfm_Yregion.groupby("time.year").mean()

num_jfm_Yregion = np.array(jfm_Yregion_by_year.to_array())[0]

reshaped_Yregion = np.reshape(num_jfm_Yregion, newshape=(
    num_jfm_Yregion.shape[0], num_jfm_Yregion.shape[1]*num_jfm_Yregion.shape[2]))

# Consider cases where group sizes are not further apart than 10 grid boxes
# ------------------------------------------------------------------------
if abs(reshaped_Xregion.shape[1]-reshaped_Yregion.shape[1]) < 10:

    # GAUSSIAN KERNEL SMOOTHING
    # -------------------------
    for var in range(reshaped_Xregion.shape[1]):
        reshaped_Xregion[:, var] = pp.smooth(reshaped_Xregion[:, var], smooth_width=12 * 10, kernel='gaussian', mask=None,
                                             residuals=True)
    for var in range(reshaped_Yregion.shape[1]):
        reshaped_Yregion[:, var] = pp.smooth(reshaped_Yregion[:, var], smooth_width=12 * 10, kernel='gaussian', mask=None,
                                             residuals=True)
    #######

    def shift_by_one(array1, array2, t):
        if t == 0:
            return array1, array2
        elif t < 0:
            s = -t
            newarray1 = array1[:-s, :]
            newarray2 = array2[s:, :]
            return newarray1, newarray2

        else:
            newarray1 = array1[t:, :]
            newarray2 = array2
            return newarray1, newarray2

    shifted_Yregion, shifted_Xregion = shift_by_one(
        reshaped_Yregion, reshaped_Xregion, 1)

    print(shifted_Xregion.shape)
    print(shifted_Yregion.shape)

    np.save(
        f'./enso_data_{i}{j}{k}{l}_X.npy', shifted_Xregion)
    np.save(
        f'./enso_data_{i}{j}{k}{l}_Y.npy', shifted_Yregion)

    cov = np.cov(np.hstack([shifted_Xregion, shifted_Yregion]).T)
    plt.imshow(cov)
    plt.colorbar()
    plt.savefig('cov.png')
    plt.close()
    # np.save('cov.npy', np.round(cov, 2))

    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cov,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.1f',
                     annot_kws={'size': 12},
                     cmap='coolwarm')
    # yticklabels=cols,
    # xticklabels=cols)
    plt.title('Covariance matrix', size=18)
    plt.tight_layout()
    plt.savefig('cov2.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.plot(ond_Xregion_by_year.year, shifted_Xregion)
    plt.title('ENSO region')
    plt.xlabel('Year')
    plt.ylabel('Temperature deviation')
    plt.savefig('ENSO.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.plot(jfm_Yregion_by_year.year[1:], shifted_Yregion)
    plt.title('BCT region')
    plt.xlabel('Year')
    plt.ylabel('Temperature deviation')
    plt.savefig('BCT.png')
    plt.close()
