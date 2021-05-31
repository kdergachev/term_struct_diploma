import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np



def str_to_list(s):

    s = re.sub("[\[\]\\n,]", " ", s).split()
    s = [float(i) for i in s]
    return s


"""
temptbl = pd.read_csv("D:\\dipl\\results\\random_points_price_NSS_w\\result")
temptbl["Date"] = pd.to_datetime(temptbl["Date"], format="%y%m%d")
temptbl["Tenors"] = temptbl["Tenors"].apply(str_to_list)
temptbl["Theta"] = temptbl["Theta"].apply(str_to_list)
"""


def total_failure(table):

    mask = table.apply(lambda x: True if np.isinf(x["Theta"][-1]) else False, 1)
    table.loc[mask, ["Bias", "Loss_P", "Loss_YTM", "R2_P", "R2_YTM", "RMSE_P", "RMSE_YTM"]] = np.full(7, np.nan)
    return table

temptbl = total_failure(temptbl)

outlist = ["random_points_price_NSS_w", "past_points_price_NSS_w", "DL_points_price_NSS_w"]
outdict = {}
for i in outlist:
    temp = pd.read_csv("D:\\dipl\\results\\" + i + "\\result")
    temp["Date"] = pd.to_datetime(temp["Date"], format="%y%m%d")
    temp["Tenors"] = temp["Tenors"].apply(str_to_list)
    temp["Theta"] = temp["Theta"].apply(str_to_list)
    # match the first word before the _
    j = re.match("([a-zA-Z]*)_", i).group(1)
    outdict[j] = temp


def plot_parameter(table, param, over="Algorithm", over2="Test", bounds=None):


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

#plot_parameter(temptbl, "RMSE_P")
#plot_parameter(temptbl, "Time")


os.chdir("D:/dipl/used_in_paper/plot_params")
o = ["Train", "Test"]
p = ["RMSE_P", "Time"]
for i in outdict.keys():
    fig, ax = plt.subplots(2, 1)
    for j, pp in enumerate(p):
        ax[j] = plot_parameter_two(outdict[i], pp,
                                           ax[j], over="Algorithm", over2="Test", bounds=False)
        ax[j].set_title(pp)
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(i)


def check_thetas(table):

    return sum(table.apply(lambda x: ~(x["Theta"][2] > 0) & (x["Theta"][4] > 0) & (x["Theta"][0] + x["Theta"][1] >= 0), 1))





def plot_tenors(table):

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
samples_out = plot_tenors(temptbl)
samples_out.savefig("sample_size", format="pdf")



def get_tabular_data(table, what, over="Train", _over="Algorithm"):

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
               "stdev": round(math.sqrt(table.loc[table[_over] == i, what].var()), 2)}
        row = pd.DataFrame(row, index=[i])
        outtb = outtb.append(row)
    return outtb


# get_tabular_data(temptbl, "RMSE_P", "Train")

os.chdir("D:/dipl/used_in_paper/tables")
ovlist = ["RMSE_P", "R2_P", "Bias", "Time", "RMSE_P_L", "RMSE_P_M", "RMSE_P_S"]
for k, v in outdict.items():
    for i in ovlist:
        df = get_tabular_data(v, i, "Test")
        df.to_latex(k + "_Test_" + i + ".txt")
ii = get_tabular_data(outdict["past"], "RMSE_P")
jj = get_tabular_data(testtb, "Time")
s = ii - jj


from statsmodels.tsa.statespace.sarimax import SARIMAX


def compare_points(table_past, table_random, algo, over="Train", by="RMSE_P"):

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




os.chdir(r"D:\dipl\results\past_points_price")
for i in os.listdir():
    j = r"D:\dipl\results\past_points_price\\" + i + "\\St_points.npy"
    print(np.load(j))



temp = testtbl.loc[(testtbl["Data"] == "Combined"), :]
temp = temp.sort_values("Loss_P", ascending=True)
temp = temp.drop_duplicates("Date")




from optimization import *
os.chdir(r"D:\dipl\used_in_paper")





def sigmafun(s, mean):
    return (0.4*s)/(1 - abs(0.5 - mean))

x = np.linspace(0, 1, 1000)

fig, ax = plt.subplots()
ax.plot(x, sigmafun(1, x))
ax.set_ylabel('scale')
ax.set_xlabel('mean')
fig.savefig("scale_truncnorm")


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








def plot_thetas(table, by="RMSE_P"):
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

plot_thetas(testtbl)








ttttt = testtbl.loc[testtbl.apply(lambda x: True if x["Theta"][0] + x["Theta"][1] < 0.001 else False, 1), :]









