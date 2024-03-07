import pandas as pd
import urllib.request
import gzip

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model

from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian
from dag_gflownet.utils.sampling import sample_from_linear_gaussian


def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, 'wb') as f:
        f.write(file_content)
    
    return filename


def get_data(name, args, rng=default_rng()):
    if name == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
            rng=rng
        )
        score = 'bge'

    elif name == 'sachs_continuous':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data/sachs.data.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name =='sachs_interventional':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
            Path('data/sachs.interventional.txt')
        )
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bde'

    elif name =='causal_BH_ell':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_ell.csv')
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name =='causal_BH_ell_half':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_ell.csv')
        # split the data randomly into half
        data = data.sample(frac=0.5,random_state=1,replace=False)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name =='causal_BH_ell_other_half':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_ell.csv')
        # split the data randomly into half
        data_1 = data.sample(frac=0.5,random_state=1,replace=False)
        data_2 = data.drop(data_1.index)
        data = data_2
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name =='causal_BH_spr':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_spr.csv')
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'
    
    elif name =='causal_BH_spr_sph':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_spr_sph.csv')
        data = (data - data.mean()) / data.std()
        score = 'bge'

    elif name =='causal_BH_len':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_len.csv')
        data = (data - data.mean()) / data.std()
        score = 'bge'

    elif name =='causal_BH_full':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_full.csv')
        data = (data - data.mean()) / data.std()
        score = 'bge'

    elif name =='hello':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/hello.csv')
        data = (data - data.mean()) / data.std()
        score = 'bge'
    
    elif name =='yashar_dataset':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/yashar_dataset.csv',delimiter='\t')
        data = (data - data.mean()) / data.std()
        score = 'bge'

    ## bootstrap
    elif name =='causal_BH_ell_bootstrap':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_ell.csv')
        # bootstrap
        data = data.sample(frac=1,random_state=args.bootstrap_seed,replace=True)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'
    
    elif name =='causal_BH_len_bootstrap':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_len.csv')
        # bootstrap
        data = data.sample(frac=1,random_state=args.bootstrap_seed,replace=True)
        data = (data - data.mean()) / data.std()
        score = 'bge'

    elif name =='causal_BH_spr_bootstrap':
        graph = ''
        data = pd.read_csv('/home/zehao/causal/jax-dag-gflownet/causal_BH_spr.csv')
        # bootstrap
        data = data.sample(frac=1,random_state=args.bootstrap_seed,replace=True)
        data = (data - data.mean()) / data.std()
        score = 'bge'

    elif name =='causal_BH_ell_LOO':
        graph = ''
        data = pd.read_csv('/home/zj448/causal/jax-dag-gflownet/causal_BH_ell.csv')
        # Leave-one-out cross-validation, remove one sample based on the LOOindex
        data = data.drop(data.index[args.LOOindex])
        data = (data - data.mean()) / data.std()
        score = 'bge'

    else:
        raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score
