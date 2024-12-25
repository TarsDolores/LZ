# -*- coding: UTF-8 -*-
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def save_np2nc(label_pred, file_path, file_name):
    """
    将np数据保存为nc文件（带经纬度坐标）
    In：
        label_pred(float) : shape为(lat,lon), CO2通量预测数据
        file_path(str): 保存的nc文件路径
        file_name(str): 保存的nc文件命名(带后缀)
    Out：
        nc文件
    """
    # 1.定义经纬度坐标范围
    lon = np.arange(-180.0, 180.0, 2.5)
    lat = np.arange(-90.0, 92.0, 2)
    # 2.
    lat_attr = dict(standard_name="lat", long_name="Latitude", units="degrees_north", axis="Y")
    lon_attr = dict(standard_name="lon", long_name="Longitude", units="degrees_east", axis="X")
    # 3.创建文件
    ds = xr.Dataset(
        {
            "EmisCO2": (["lat", "lon"], label_pred)
        },
        coords={
            "lat": (["lat"], lat, lat_attr),
            "lon": (["lon"], lon, lon_attr),
        },
    )
    # 4.保存文件
    nc_file = file_path + file_name + '.nc'
    ds.to_netcdf(nc_file)


def draw_nc2png(nc_value, file_path, file_name):
    """
    将带经纬度坐标的nc文件绘图
    In：
        nc_value(float): shape为(91,144), CO2通量数据
    Out：
        img
    """
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.ax = ax

    m.drawcoastlines(linewidth=0.75)
    m.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0], fontsize=15, linewidth=0.5)
    m.drawmeridians(np.arange(0, 360, 60), labels=[0, 0, 0, 1], fontsize=15, linewidth=0.5)

    cmap = plt.get_cmap('RdBu_r')
    clevs = np.arange(-15, 15.5, 1.5)

    cs = nc_value.plot.contourf(ax=ax, cmap=cmap, levels=clevs, add_labels=False, add_colorbar=False, zorder=0, extend='both')
    fig.colorbar(cs, ax=ax)

    png_file = file_path + file_name+'.png'
    plt.savefig(png_file, dpi=200)
    plt.close()
    return png_file


def plot_corr(corr, file_png):
    fig, ax = plt.subplots(figsize=(45, 5))
    ax.plot(corr, marker='o', markersize=4)
    ax.set_ylabel('corr')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(corr))
    plt.savefig(file_png, dpi=300)
    plt.close()


