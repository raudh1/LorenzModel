def app_entropy(x, order=2, metric='chebyshev'):
    
    phi = _app_samp_entropy(x, order=order, metric=metric, approximate=True)    # it defines the vector phi
    return np.subtract(phi[0], phi[1])

def _app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`.
        """
    #technical stuff
    
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))



    phi = np.zeros(2) #vector phi
    r = 0.2 * np.std(x, ddof=0) #radium to define the concept of neighbourds (0.2* standard deviation with 0 DF of the timeseries)
    
    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1) #creates a matrix with the timeseries Nx2
    if approximate: # TRUE
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True
                                                           ).astype(np.float64)
                                                           # compute phi(order + 1, r)
emb_data2 = _embed(x, order + 1, 1) # same matrix but Nx3
count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                       count_only=True
                                                       ).astype(np.float64)            # count the number of neighbours according to Chebyshev norm for
                                                                                       # each point and put this number in a vector.

if approximate: #true
    phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))                              
    phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi






def _embed(x, order=3, delay=1):
    
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[(i * delay):(i * delay + Y.shape[1])]
    return Y.T


@jit('UniTuple(float64, 2)(float64[:], float64[:])', nopython=True)
