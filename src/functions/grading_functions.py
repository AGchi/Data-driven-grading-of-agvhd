import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


# calculate p_0 which is needed for calculating severity score
def calc_p_0(gvhd_data, skin_grade, liver_grade, git_grade):
    skin_mean = np.mean(gvhd_data[skin_grade])
    liver_mean = np.mean(gvhd_data[liver_grade])
    intestinal_mean = np.mean(gvhd_data[git_grade])
    p_0_list = [skin_mean, liver_mean, intestinal_mean]
    return p_0_list


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se*scipy.stats.t.ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

# function to create PCA biplot
def myplot_biplot(score, coeff, xmin, xmax, ymin, ymax, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max()-xs.min())
    scaley = 1.0/(ys.max()-ys.min())
    plt.scatter(xs*scalex, ys*scaley, c='#a6cee3')
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='#386cb0', alpha=0.5)
        if labels[i] == "GI":
            plt.text(coeff[i, 0]*1.06, coeff[i, 1]*1.05, labels[i],
                     color='black', ha='center', va='center', fontsize=11)
        if labels[i] == "Skin":
            plt.text(coeff[i, 0]*1.15, coeff[i, 1]*1.05, labels[i],
                     color='black', ha='center', va='center', fontsize=11)
        if labels[i] == "Liver":
            plt.text(coeff[i, 0]*1.23, coeff[i, 1]*1.45, labels[i],
                     color='black', ha='center', va='center', fontsize=11)
        if labels is None:
            plt.text(coeff[i, 0]*1.15, coeff[i, 1]*1.15, "Var"+str(i+1),
                     color='black', ha='center', va='center', fontsize=11)
    # else:
    #     plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i],
    #     color = 'black', ha = 'center', va = 'center', fontsize=11)

    # else:
    #     plt.text(coeff[i,0]* 1.15, coeff[i,1]* 1.05 , labels[i],
    #     color = 'black', ha = 'center', va = 'center', fontsize=11)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("PC{}".format(1), fontsize=15)
    plt.ylabel("PC{}".format(2), fontsize=15)
    plt.title("Biplot", fontsize=16)
    plt.grid()


# functions to calculate PC based gvhd stages

def calc_gvhd_score(gvhd_data, p0, c1, skin_grade, liver_grade, git_grade,
                    colname):
    skin = gvhd_data[skin_grade]
    liver = gvhd_data[liver_grade]
    intestinal = gvhd_data[git_grade]
    s_strich_i_list = []
    for s, l, i in zip(skin, liver, intestinal):
        vector = [s, l, i]
        s_strich_i_list.append((np.dot((np.subtract(vector, p0)), c1)).item())
    si_strich_no_shift_list = [round(2*i) for i in s_strich_i_list]
    if min(si_strich_no_shift_list) < 0:
        s0_p_shifted_list = [x+abs(min(si_strich_no_shift_list))+1 for x in
                             si_strich_no_shift_list]
    elif min(si_strich_no_shift_list) == 0:
        s0_p_shifted_list = [x+1 for x in si_strich_no_shift_list]
    elif min(si_strich_no_shift_list) == 1:
        s0_p_shifted_list = si_strich_no_shift_list
    elif min(si_strich_no_shift_list) > 1:
        s0_p_shifted_list = [x-abs(min(si_strich_no_shift_list))+1 for x in
                             si_strich_no_shift_list]
    gvhd_data[colname] = s0_p_shifted_list
    return gvhd_data


def calc_gvhd_score_simplified(gvhd_data, p0, c1, skin_grade, liver_grade,
                               git_grade, colname):
    skin = gvhd_data[skin_grade]
    liver = gvhd_data[liver_grade]
    intestinal = gvhd_data[git_grade]
    s_strich_i_list = []
    for s, l, i in zip(skin, liver, intestinal):
        vector = [s, l, i]
        s_strich_i_list.append((np.dot((np.subtract(vector, p0)), c1)).item())
    si_strich_no_shift_list = [round(i) for i in s_strich_i_list]
    if min(si_strich_no_shift_list) < 0:
        s0_p_shifted_list = [x+abs(min(si_strich_no_shift_list))+1 for x in
                             si_strich_no_shift_list]
    elif min(si_strich_no_shift_list) == 0:
        s0_p_shifted_list = [x+1 for x in si_strich_no_shift_list]
    elif min(si_strich_no_shift_list) == 1:
        s0_p_shifted_list = si_strich_no_shift_list
    elif min(si_strich_no_shift_list) > 1:
        s0_p_shifted_list = [x-abs(min(si_strich_no_shift_list))+1 for x in
                             si_strich_no_shift_list]
    gvhd_data[colname] = s0_p_shifted_list
    return gvhd_data

