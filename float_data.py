import fsspec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import pickle
import cartopy

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from matplotlib.dates import DateFormatter
from matplotlib.pyplot import cm
from scipy import signal
import scipy.stats as sts


def retrieve_profiles(timeRange = slice('1965-01', '1994-12'), levSel=slice(100, 2000), maxLat = -30, mask = None, options = {}):
  """
  Create an xarray.DataArray of the temperature-depth profiles in the Southern Ocean. 
  
  :param timeRange: a slice or list of the times 
  :param levSel: a slice or list of the depths 
  :param maxLat: the maximum latitude
  :param mask: a list of stacked ('i', 'j') coordinates to include, e.g. to remove NaN values
  :param options: dictionary to select the ensemble. This function contains an additional option 'raw' which, if true, will cause
  this function to return unstacked data. In this case, the parameter mask is ignored
  :return: xarray.DataArray with coordinates ['time', 'n', 'lev'] (['time', 'lat', 'lon', 'lev'] if options['raw'])
  
  """
  # options contains all the information to identify the data
  options_default = {
      'dataVariableId' : 'thetao',
      'dataSourceId' : 'UKESM1-0-LL',
      'experimentId' : 'historical',
      'tableId' : 'Omon',
      'memberId' : 'r1i1p1f2',
      'raw' : False
  }
  options = {**options_default, **options}

  # take the profiles from wherever
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

  dataRaw = fileSet.thetao

  try:
      dataRaw = dataRaw.rename({"latitude":"lat", "longitude":"lon"})
  except:
      pass

  data = dataRaw.sel(lev=levSel, time=timeRange)
  data = data.where(data.lat < maxLat, drop=True)
  #data = data.squeeze()
  if options['raw']:
    return data
  data = data.stack(n=('i', 'j',))

  # mask is a list of the values of n that are not NA
  if mask is None:
    mask = data.isel(time=0).dropna('n')['n'].values
  data = data.sel(n=mask)

  return data


def random_sample(data, N):
  """
  Returns a random sample of the data. The input array should have dimensions 
  time, n and lev. Returns an array with N*len(data['time']) profiles, evenly sampled
  over time. Use data.persist() or data.compute() after this to avoid resampling for every computation
  
  :param data: the temperature-depth profiles, coordinates ['time', 'n', 'lev']
  :param N: the number of samples from each time 
  :return: xarray.DataArray with coordinates ['N', 'lev']
  """
  rng = np.random.default_rng()
  def func(arr):
    return rng.choice(arr, N, replace=False)

  result = xr.apply_ufunc(
        func,
        data,
        input_core_dims=[['n', 'lev']],
        output_core_dims=[['M', 'lev']],
        dask='parallelized',
        output_dtypes=('float64',),
        vectorize=True,
        dask_gufunc_kwargs={
            'output_sizes' : {'M' : N}
        }
        
    )
  
  result = result.stack(N=('M', 'time'))
  result = result.transpose('N', 'lev')
  return result


def normalise_data(data, dim=-1, return_inv=False):
  if return_inv:
    mean = data.mean(dim)
    std = data.std(dim)
    return ((data - mean) / std, mean, std)
  else:
    return (data - data.mean(dim)) / data.std(dim)

def train_pca(data_sampled, n_components=3):
  """
  data_sampled (N, lev) returns pca object
  """
  
  pca = PCA(n_components)
  arr = data_sampled.values
  pca = pca.fit(arr)
  return pca

def pca_transform(data, pca):
  """
  Applies a transformation into PCA space
  
  """

  lev_size = data.sizes['lev']
  n_comp = pca.n_components
  
  def func(arr):
    arr_r = np.reshape(arr, (-1, lev_size))
    
    inds = np.isnan(arr_r)
    arr_r[inds] = 0
    
    out = pca.transform(arr_r)
    out[inds[..., 0:n_comp]] = np.nan
    out_sizes = np.shape(arr[..., 0:n_comp])
    
    out = np.reshape(out, out_sizes)
    return out

  result = xr.apply_ufunc(
        func,
        data,
        input_core_dims=[['lev']],
        output_core_dims=[['pca_comp']],
        dask='parallelized',
        output_dtypes=('float64',),
        vectorize=False,
        dask_gufunc_kwargs={
          'output_sizes' : {'pca_comp' : n_comp},
          'allow_rechunk' : True
        }
        
    )
  
  return result

def train_gmm(data_trans, K):
  """
  Trains a GMM on the data
  """
  gmm = GaussianMixture(K)
  gmm.fit(data_trans.values)
  return gmm

def gmm_classify(data_trans, gmm):
  
  """
  Replace the nan with -1
  """

  pca_size = data_trans.sizes['pca_comp']
  def func(arr):
    arr_r = np.reshape(arr, (-1, pca_size))
    
    inds = np.isnan(arr_r) 
    arr_r[inds] = 0
    
    out = gmm.predict(arr_r)
    out_sizes = np.shape(arr[..., 0])
    out[inds[:, 0]] = -1 #replaces the nan values with -1
    out = np.reshape(out, out_sizes)
    return out

  result = xr.apply_ufunc(
        func,
        data_trans,
        input_core_dims=[['pca_comp']],
        output_core_dims=[[]],
        dask='parallelized',
        output_dtypes=('int',),
        vectorize=False,
    )
  
  return result

def gmm_prob(data_trans, gmm):
  
  """
  Replace the nan with -1
  """

  pca_size = data_trans.sizes['pca_comp']
  gmm_size = gmm.n_components
  def func(arr):
    arr_r = np.reshape(arr, (-1, pca_size))
    
    inds = np.isnan(arr_r) 
    arr_r[inds] = 0
    
    out = gmm.predict_proba(arr_r)
    out_sizes = list(np.shape(arr))
    out_sizes[-1] = gmm_size
    out[inds[:, 0]] = np.nan
    out = np.reshape(out, out_sizes)
    return out

  result = xr.apply_ufunc(
        func,
        data_trans,
        input_core_dims=[['pca_comp']],
        output_core_dims=[['k']],
        dask='parallelized',
        output_dtypes=('float64',),
        vectorize=False,
        dask_gufunc_kwargs={
            'output_sizes' : {'k' : gmm_size}
        }
    )
  
  return result
  
  
def gmm_score_samples(data_trans, gmm):
  
  """
  Replace the nan with -1
  """

  pca_size = data_trans.sizes['pca_comp']
  gmm_size = gmm.n_components
  def func(arr):
    arr_r = np.reshape(arr, (-1, pca_size))
    
    inds = np.isnan(arr_r) 
    arr_r[inds] = 0
    
    out = gmm.score_samples(arr_r)
    out_sizes = np.shape(arr)
    out_sizes = out_sizes[0:-1]
    out[inds[:, 0]] = np.nan
    out = np.reshape(out, out_sizes)
    return out

  result = xr.apply_ufunc(
        func,
        data_trans,
        input_core_dims=[['pca_comp']],
        output_core_dims=[[]],
        dask='parallelized',
        output_dtypes=('float64',),
        vectorize=False,
        dask_gufunc_kwargs={
            'output_sizes' : {}
        }
    )
  
  return result

def count_area(data_classes, K):
  """
  Counts the number of assignments for each class, weighted by the latitude. The result is proportional to the total ocean surface area
  of profiles assigned to each class
  """
  lats = data_classes['lat'].values
  cos_lats = np.cos(lats * np.pi / 180)
  def func(arr):
    out = np.zeros(K)
    for i in range(K):
      out[i] = np.sum(cos_lats * (arr==i))
    return out

  result = xr.apply_ufunc(
        func,
        data_classes,
        input_core_dims=[['n']],
        output_core_dims=[['k']],
        dask='parallelized',
        output_dtypes=('float64',),
        vectorize=True,
        dask_gufunc_kwargs={
            'output_sizes' : {'k' : K}
        }
    )
  
  return result

def mean_lat(data_classes, K):
  lats = data_classes['lat'].values
  # cos_lats = np.cos(lats * np.pi / 180)
  def func(arr):
    out = np.zeros(K)
    for i in range(K):
      out[i] = np.mean(lats * (arr==i))
    return out

  result = xr.apply_ufunc(
        func,
        data_classes,
        input_core_dims=[['n']],
        output_core_dims=[['k']],
        dask='parallelized',
        output_dtypes=('float64',),
        vectorize=True,
        dask_gufunc_kwargs={
            'output_sizes' : {'k' : K}
        }
    )
  
  return result


def avg_profiles(data, data_classes, K):
  """
  Returns some information about the average properties of the profiles
  mean
  std
  returns a list of dictionaries
  """

  out = []
  for i in range(K):
    d = data.where(data_classes==i)
    d_m = d.mean(('time', 'n'), skipna=True).values
    d_std = d.std(('time', 'n'), skipna=True, ddof=1).values
    out.append({'mean' : d_m, 'std' : d_std})
  return out

def pca_sort(data_classes, gmm):
  """
  Returns a new array of class assignnments, with the classes now ordered by mean value of the first pca component
  """
  means = gmm.means_[:, 0]
  inds = np.argsort(means)
  return reorder(data_classes, inds)


def match_profiles(avg0, avg1, dist=None):
  """
  Attempt to match profiles of the same shape
  avg0: list of dict, profile means and std
  avg1: list of dict, profile means and std
  dist: dict, dict -> float64 distance between two profiles
  returns:
  inds: array of indices that sorts the profiles in avg1 to match avg0
  """
  n_classes = len(avg0)
  assert n_classes == len(avg1)
  norms = np.zeros((n_classes, n_classes))
  for i in range(n_classes):
    for j in range(n_classes):
      if dist is None:
        diff = avg0[i]['mean'] - avg1[j]['mean']
        norms[i, j] = np.sum(diff * diff)
        #norms[j, i] = norms[i, j]
      else:
        norms[i, j] = dist(avg0[i], avg1[j])
        #norms[j, i] = norms[i, j]
  return np.argmin(norms, axis=1)

def match_spatial(data_classes, data_classes_ref, n_classes):
  """
  Attempt to match profiles with the same spatial distribution
  data_classes: 
  data_classes_ref:
  n_classes
  returns:
  inds: array of indices that sorts the classes in data_classes
  """
  
  counts = np.array([[((data_classes.where(data_classes_ref==k) == j) * np.cos(data_classes['lat'] * 3.1415/180)).sum(skipna=True) for k in range(n_classes) ] for j in range(n_classes)])
  return np.argmax(np.array(counts), axis=0)


def temp_sort(data_classes, avg, arg=False):
  """
  Returns a new array of class assignnments, with the classes now ordered by mean value of the first pca component
  """
  top_mean = np.array([i['mean'][0] for i in avg])
  inds = np.argsort(top_mean)
  if arg:
    return inds
  else:
    return reorder(data_classes, inds)
  
def reorder(data_classes, inds):
  inds_r = np.zeros(np.shape(inds))
  inds_r[inds] = np.arange(0, np.size(inds), dtype='int')
  
  def func(arr):
    out = inds_r[arr.astype('int')]
    out[arr == -1] = -1
    return out

  result = xr.apply_ufunc(
        func,
        data_classes,
        input_core_dims=[[]],
        output_core_dims=[[]],
        dask='parallelized',
        output_dtypes=('int',),
        vectorize=False,
        dask_gufunc_kwargs={
            'output_sizes' : {}
        }
    )
  
  return result

def modal_classes(data_classes):
  
  def func(arr):
    return sts.mode(arr)[0]

  return xr.apply_ufunc(
        func,
        data_classes,
        input_core_dims=[['time']],
        output_core_dims=[[]],
        dask='parallelized',
        output_dtypes=('float64',),
        vectorize=True,
        dask_gufunc_kwargs={
          'output_sizes' : {},
          'allow_rechunk' : True
        }
        
    )
  
  