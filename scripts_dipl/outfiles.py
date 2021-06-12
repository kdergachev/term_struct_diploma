import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


def str_to_list(s):
    """
    Used in apply to turn string from .csv to list

    """

    s = re.sub("[\[\]\\n,]", " ", s).split()
    s = [float(i) for i in s]
    return s


# Get the three result tables in a dict
outlist = ["random_points_price_NSS_w_double", "past_points_price_NSS_w_double", "DL_points_price_NSS_w_double"]
outdict = {}
for i in outlist:
    temp = pd.read_csv("D:\\dipl\\results\\" + i + "\\result")
    temp["Date"] = pd.to_datetime(temp["Date"], format="%y%m%d")
    temp["Tenors"] = temp["Tenors"].apply(str_to_list)
    temp["Theta"] = temp["Theta"].apply(str_to_list)
    # match the first word before the _
    j = re.match("([a-zA-Z]*)_", i).group(1)
    outdict[j] = temp


# Here plots used in the crisis test were generated
def plot_parameter(table, param, over="Algorithm", over2="Test", bounds=None):
    """
    Plot results in param column for different values in over gathered by over2 parameter
    i.e. RMSE for each algorithm over test dataset

    """

    fig, ax = plt.subplots()
    for i in table[over].unique():
        out = table.loc[(table[over] == i) & (table["Data"] == over2), ["Date", param]]
        if bounds is False:
            mask = out[param] < (out[param].median() + math.sqrt(out[param].var())*0.7)
            mask = mask & (out[param] > (out[param].median() - math.sqrt(out[param].var())*0.7))
        elif bounds is not None:
            mask = (bounds[0] < out[param]) & (out[param] < bounds[1])
        else:
            mask = np.full(len(out[param]), True)
        #out = out.loc[mask, :]
        line, = ax.plot(out.loc[mask, "Date"], out.loc[mask, param], ls="--", lw=0.9)
        out.loc[~mask, :] = np.nan
        line, = ax.plot(out["Date"], out[param], marker="o", markersize=2, color=line.get_color(), lw=1)
        line.set_label(i)
    ax.legend()
    return fig


os.chdir("D:/dipl/used_in_paper/resplots")
a = plot_parameter(outdict["past"], "RMSE_P", over2="Test")
a.savefig("RMSE")
a = plot_parameter(outdict["past"], "RMSE_P_R", over2="Test")
a.savefig("RMSE_R")
a = plot_parameter(outdict["past"], "Time", over2="Test")
a.savefig("Time")
a= plot_parameter(outdict["past"], "Time_R", over2="Test")
a.savefig("Time_R")


# Get plot to show sample sizes
def plot_tenors(table):
    """
    Plot time to maturity of a dataset

    """

    tenorplot = []
    for i in table["Date"].unique():
        tenorplot.extend([(i, j) for j in table.loc[table["Date"] == i, "Tenors"].iloc[0]])
    fig, ax = plt.subplots()
    ax.scatter(*zip(*tenorplot), s=1.3)
    ax.set_ylabel('Time to maturity')
    ax.set_xlabel('Date')
    return fig


#ppppp = plot_tenors(ii)
os.chdir(r"D:\dipl\used_in_paper")
samples_out = plot_tenors(outdict["past"])
samples_out.savefig("sample_size", format="pdf")


# Make tables to use in the paper
def get_tabular_data(table, what, over="Train", _over="Algorithm"):
    """
    Organize parameter (what) in a table (using dataset in over) comparing _over with min, max, median..........

    """

    # whatlist.append(_over)
    table = table.loc[table["Data"] == over, [what] + [_over]]
    outtb = pd.DataFrame()
    for i in table[_over].unique():
        row = {"min": round(table.loc[table[_over] == i, what].min(), 2),
               "0.25": round(table.loc[table[_over] == i, what].quantile(0.25), 2),
               "median": round(table.loc[table[_over] == i, what].quantile(0.5), 2),
               "0.75": round(table.loc[table[_over] == i, what].quantile(0.75), 2),
               "max": round(table.loc[table[_over] == i, what].max(), 2),
               "mean": round(table.loc[table[_over] == i, what].mean(), 2),
               "stdev": round(math.sqrt(table.loc[table[_over] == i, what].var(skipna=True)), 2)}
        row = pd.DataFrame(row, index=[i])
        outtb = outtb.append(row)
    return outtb


os.chdir("D:/dipl/used_in_paper/tables")
ovlist = ["Time", "RMSE_P_L", "RMSE_P_M", "RMSE_P_S", "Time_R", "RMSE_P_L_R", "RMSE_P_M_R", "RMSE_P_S_R"]
for k, v in outdict.items():
    for i in ovlist:
        df = get_tabular_data(v, i, "Test")
        df.to_latex(k + "_Test_" + i + ".txt")


for k, v in outdict.items():
    df = get_tabular_data(v, "R2_P", "Combined")
    df.to_latex(k + "_Test_" + "R2" + ".txt")


from optimization import *
os.chdir(r"D:\dipl\used_in_paper")


# Plot scale of truncnorm to show in paper
def sigmafun(s, mean):
    """
    Scaling parameter function to draw later

    """
    return (0.4*s)/(1 - abs(0.5 - mean))

x = np.linspace(0, 1, 1000)

fig, ax = plt.subplots()
ax.plot(x, sigmafun(1, x))
ax.set_ylabel('scale')
ax.set_xlabel('mean')
fig.savefig("scale_truncnorm")


# Plot 4 versions of truncnorm on (0,1) to display
from optimization import uneven_distr
plt.ioff()
fig, ax = plt.subplots(2, 2)
for i in range(4):
    j = [(0, 0.5), (0, 0), (0.5, 0.5), (0.2, 0.2)][i]
    x = uneven_distr(10000, np.array([j[0]]), np.array([0]), np.array([1]))
    y = uneven_distr(10000, np.array([j[1]]), np.array([0]), np.array([1]))
    ax[i // 2, i % 2].hist2d(np.asarray(x)[:, 0], np.asarray(y)[:, 0])
    ax[i // 2, i % 2].set_title("Mean at " + str(j))

fig.savefig("truncnorm_examples")


# test for good guess

# get theta guesses for each day
from scipy import stats
import statsmodels.api as sm
thetadict = {}
for i in outlist:
    os.chdir("D:/dipl/results/" + i)
    temp = {}
    for j in os.listdir():
        if os.path.isdir(j):
            jj = datetime.strptime(j, "%y%m%d")
            temp[jj] = np.load("D:/dipl/results/" + i + "/" + j + "/Theta_best.npy")
    print(np.array(list(temp.values())))
    thetadict[re.match("([a-zA-Z]*)_", i).group(1)] = pd.DataFrame(np.array(list(temp.values())), index=temp.keys())
    thetadict[re.match("([a-zA-Z]*)_", i).group(1)] = thetadict[re.match("([a-zA-Z]*)_", i).group(1)].apply(lambda x: np.array([x[0], x[1], x[2], x[3]]), 1)

thetadict.pop("random")
thetadict["past"] = thetadict["past"].iloc[1:]

# matching dates get 2 thetas for each starting point generation method: realization and guess
res = {"past": {"Guess": [], "Realization": []}, "DL": {"Guess": [], "Realization": []}}
for k, v in thetadict.items():
    opt = outdict[k].loc[outdict[k]["Data"] == "Combined", :]
    opt = opt.sort_values("RMSE_P")
    opt = opt.drop_duplicates("Date").loc[:, ["Date", "Theta"]]
    for i in opt["Date"]:
        try:
            res[k]["Guess"].append(v[i])
            res[k]["Realization"].append(np.array(opt.loc[opt["Date"] == i, "Theta"].to_numpy()[0]))
        except:
            pass
    res[k]["Guess"] = np.array(res[k]["Guess"])
    res[k]["Realization"] = np.array(res[k]["Realization"])


def split_into_regressions(yarray, xarray):
    """
    Run the regression over each of the axis of supplied vectors, returns matrix of the form
             intercept    slope
    value        a          b
    p-value      c          d
    if constant is supplied intercept part is empty and slope is shown (a in Y = aX)
    """
    res = np.zeros((yarray.shape[1], 2, 2))
    for i in range(yarray.shape[1]):
        x = np.apply_along_axis(lambda a: a[i], 1, xarray)
        y = np.apply_along_axis(lambda a: a[i], 1, yarray)
        x = sm.add_constant(x)
        u = sm.OLS(y, sm.add_constant(x))
        o = u.fit()
        try:
            o = np.concatenate([o.params, o.pvalues]).reshape((2, 2))
        except ValueError:
            o = np.concatenate([np.append(o.params, np.nan), np.append(o.pvalues, np.nan)]).reshape((2, 2))
        res[i] = o
    return res

print(split_into_regressions(res["past"]["Realization"], res["past"]["Guess"]).round(3))
print(split_into_regressions(res["DL"]["Realization"], res["DL"]["Guess"]).round(3))



############################## Not used ######################

"""
temptbl = pd.read_csv("D:\\dipl\\results\\past_points_price_NSS_w_double\\result")
temptbl["Date"] = pd.to_datetime(temptbl["Date"], format="%y%m%d")
temptbl["Tenors"] = temptbl["Tenors"].apply(str_to_list)
temptbl["Theta"] = temptbl["Theta"].apply(str_to_list)
"""


tt = sm.OLS(res["past"]["Realization"], res["past"]["Guess"])



def total_failure(table):
    """
    If optimization failed return np.nan row
    """

    mask = table.apply(lambda x: True if np.isinf(x["Theta"][-1]) else False, 1)
    table.loc[mask, ["Bias", "Loss_P", "Loss_YTM", "R2_P", "R2_YTM", "RMSE_P", "RMSE_YTM"]] = np.full(7, np.nan)
    return table


"""
temptbl = total_failure(temptbl)
"""


def check_theta_domain(table, l, h, cons, _custom_main=False):
    """
    See if any of the thetas are outside the domain

    """
    res = table.apply(lambda x: (x["Theta"] < l).any() | (x["Theta"] > h).any() |
                                (cons(x["Theta"]) < 0), 1)
    if _custom_main:
        res = table.apply(lambda x: (x["Theta"][0] < 0) | (x["Theta"][2] <= 0) |
                                    (x["Theta"][0] + x["Theta"][1] < 0), 1)
    return res


# a = check_theta_domain(temptbl, np.array([0, -5, 0.001, -20]), np.array([10, 35, 35, 35]), cons=lambda y: y[0] + y[1],
#                        _custom_main=True)
# b = temptbl.loc[a, :]


def check_thetas(table):
    """
    Not used.
    Was a test to see if thetas are unreasonable
    """
    return sum(table.apply(lambda x: ~(x["Theta"][2] > 0) & (x["Theta"][4] > 0) & (x["Theta"][0] + x["Theta"][1] >= 0), 1))



from statsmodels.tsa.statespace.sarimax import SARIMAX


def compare_points(table_past, table_random, algo, over="Train", by="RMSE_P"):
    """
    Not used
    One idea to run regression to see if past data works

    """

    best = table_past.sort_values(by, ascending=True).drop_duplicates("Date")
    best = best.loc[:, ["Date", by]]
    best = best.sort_values("Date")
    best = best.set_index("Date")
    reg_1 = table_past.loc[(table_past["Algorithm"] == algo) & (table_past["Data"] == over), ["Date", by]]
    reg_1 = table_random.loc[(table_random["Algorithm"] == algo) & (table_random["Data"] == over), ["Date", by]] - reg_1
    reg_1 = reg_1.set_index("Date")
    best = best.shift(1)
    reg_1 = SARIMAX(reg_1.iloc[1:, :].to_numpy(), best.iloc[1:, :].to_numpy(), order=(0, 0, 0), trend='c').fit()

    return res
# L^R - L^P = c + aL^*_{t-1}
a = compare_points(testtbl, testtb, "Diff_evo")
a.params
a.pvalues
b = compare_points(testtb, "PSO")
b.params
b.pvalues




def plot_thetas(table, by="RMSE_P"):
    """
    Plot some results against different thetas

    """
    best = table.sort_values(by, ascending=True).drop_duplicates("Date")
    best = best.loc[:, ["Date", "Theta"]]
    best = best.sort_values("Date")
    best = best.set_index("Date")
    for i in range(len(best.iloc[0, 0])):
        fig, ax = plt.subplots()
        temp = best.apply(lambda x: x.item()[i], 1)
        if i == 2 or i == 4:
            print(sum(temp >= 29))
        ax.plot(temp)
        fig.show()
    return fig

plot_thetas(temptbl)


