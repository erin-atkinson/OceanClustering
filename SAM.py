import dask.dataframe as dd
import fsspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from dask import delayed
from scipy import signal

def retrieve_SAM(latSel = (-65, -40), timeRange = None, options = {}):
  """
  Calculate the SAM index for the first ensemble that matches the parameters in options. Return an xarray.DataArray containing the
  SAM index normalised with respect to the time range given.

  :param latSel: tuple of the two latitudes used to determine the SAM index. Default (-65, -40)
  :param timeRange: a slice, the time over which the SAM is calculated. Default None (uses all the available data)
  :param options: dictionary to select the ensemble.
  :return: DataArray with coordinates ['time']
  """ 

  # options contains all the information to identify the data
  options_default = {
      'dataVariableId' : 'psl',
      'dataSourceId' : 'UKESM1-0-LL',
      'experimentId' : 'historical',
      'tableId' : 'Amon',
      'memberId' : 'r1i1p1f2',
  }
  options = {**options_default, **options}

  df = pd.read_csv('https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv')
  dfFilt = df[df.variable_id.eq(options['dataVariableId']) 
  & df.source_id.eq(options['dataSourceId']) 
  & df.table_id.eq(options['tableId']) 
  & df.experiment_id.eq(options['experimentId']) 
  & df.member_id.eq(options['memberId'])]
  filesRaw = [xr.open_zarr(fsspec.get_mapper(item), consolidated=True) for item in dfFilt.zstore.values]
 

  for (i, v) in enumerate(filesRaw): #Formatting dates into np.datetime64 format
    startDateIterate = np.datetime64(v['time'].values[0],'M')
    endDateIterate = np.datetime64(v['time'].values[-1],'M') + np.timedelta64(1,'M')
    v['time']=('time', np.arange(startDateIterate, endDateIterate, dtype='datetime64[M]'))
    v['time_bnds']=('time_bnds', np.arange(startDateIterate, endDateIterate, dtype='datetime64[M]')) 
  fileSet = filesRaw[0] # just use one


  pressureA = fileSet.psl.sel(lat=latSel, method='nearest') #Selection of latitude
  pressureA = pressureA.reset_coords(drop=True) #Removes lev if single value
  pressureA = pressureA.mean(["lon"], keep_attrs=False)

  if timeRange is not None:
    pressureA = pressureA.sel(time=timeRange)
  # Here we compute some things
  pressureAmean = pressureA.mean(["time"])
  pressureAstd = pressureA.std(['time'])
  pressureA = (pressureA - pressureAmean) / pressureAstd
  # The isel is not necessary
  pressureA = pressureA.diff('lat').isel(lat=0).reset_coords(drop=True)
  return pressureA

def BW_filter(data, order, f, dim, btype='low'):
  """
  Apply a butterworth filter, forwards and backwards, to an xarray.DataArray
  
  :param data: the xarray.DataArray containing the data to be filters
  :param order: the order of the filter
  :param f: the cutoff frequency of the filter
  :param dim: the dimension name over which to filter the data
  :return:  DataArray with coordinates [..., dim], the filtered data
  """
  
  sos = signal.butter(order, f, output='sos', btype=btype)
  
  def func(arr):
    return signal.sosfiltfilt(sos, arr, axis=-1)

  result = xr.apply_ufunc(
        func,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask='parallelized',
        output_dtypes=('float64',),
        vectorize=False,
        dask_gufunc_kwargs={
          'output_sizes' : {dim : data.sizes[dim]},
          'allow_rechunk' : True
        }
        
    )
  
  return result

def monthly_mean(data):
  monthly = data.groupby('time.month') - data.groupby('time.month').mean('time')
  monthly = monthly.chunk('auto')
  return monthly

def load_obvs(filename):
  """
  Load observational SAM data (eg. http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.txt) from a text file.
  :param filename: the path to the data (remove the top three lines from the .txt in the link)
  :return: xarray.DataArray with coordinates ['time'] containing the data 
  """
  arr = np.genfromtxt(filename)
  yrs = arr[:, 0].astype(int)
  data = arr[:, 1:].flatten()
  t = np.arange(np.datetime64('{}-01'.format(yrs[0])), np.datetime64('{}-01'.format(yrs[-1]+1)), dtype='datetime64[M]')
  da = xr.DataArray(
    data=data,
    dims=["time"],

    coords=dict(
        time=t,
    ),

    attrs=dict(
        description="Observational SAM index",
    ),
    )
  return da


  
  


  
def moving_mean(data, n, dim):
  """
  Moving mean of the data
  
  """

  n_dim = data.sizes[dim]
  
  def func(a) :
    ret = np.cumsum(a, axis=-1)
    ret[..., n:] = ret[..., n:] - ret[..., :-n]
    ret[..., 0:n - 1] = (ret[..., 0:n - 1] * n) / np.arange(1, n)
    return ret[..., n - 1:] / n

  result = xr.apply_ufunc(
        func,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask='parallelized',
        output_dtypes=('float64',),
        vectorize=False,
        dask_gufunc_kwargs={
          'output_sizes' : {dim : n_dim},
          'allow_rechunk' : True
        }
        
    )
  
  return result