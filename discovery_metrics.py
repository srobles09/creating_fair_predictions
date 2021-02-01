# Discovery Metrics
from scipy.stats import ttest_ind

def get_difference_means_test(data1, data2):
    stat, p = ttest_ind(data1, data2)
    #print('t=%.3f, p=%.3f' % (stat, p))
    return stat,p
    

