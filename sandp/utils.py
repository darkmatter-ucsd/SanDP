from root_numpy import root2array
import pandas as pd
import numpy as np

def load_dataframe(filename):
    '''
    convert ROOT file into dataframe format.
    :param filename: file name including the path
    :return data: data in pandas dataframe format
    '''
    data_1 = pd.DataFrame(root2array(filename, 'T1'))

    array_branches = []
    scalar_branches = ['BaseLineChannel', 'BaseLineChannelSigma', 'S2sPMT']  # these two are not scalar
    for name in data_1.columns.values:
        if name in scalar_branches:
            continue
        if hasattr(data_1[name][0], '__len__'):
            array_branches.append((name, np.nan))
            #TODO add length to different array
        else:
            scalar_branches.append(name)

    data_2 = pd.DataFrame(root2array(filename, branches=array_branches))

    data = pd.concat([data_1[scalar_branches], data_2], axis=1)
    
    # add alias
    data['drift_time'] = (data['S2sPeak'] - data['S1sPeak']) * 4 / 1000  # us
    data['r'] = np.sqrt(data['S2sPosX']**2 + data['S2sPosY']**2)
    
    # convert unit
    data['S2sLowWidth'] = data['S2sLowWidth'] * 4 / 1000  # us
    data['S2sWidth'] = data['S2sWidth'] * 4 / 1000  # us
    data['S2sPeak'] *= 4/1000  #  us
    data['S2sRiseTime'] *= 4/1000  #  us
    data['S2sDropTime'] *= 4/1000  #  us
    
    data['S1sPeak'] *= 4  #  ns
    data['S1sWidth'] *= 4  #  ns
    data['S1sLowWidth'] *= 4  #  ns
    data['S1sRiseTime'] *= 4  #  ns
    data['S1sDropTime'] *= 4  #  ns
    
    return data
