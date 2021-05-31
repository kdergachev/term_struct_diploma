# -*- coding: utf-8 -*-
import datetime

import matplotlib.pyplot as plt
import pandas as pd

from table_prep import *
from optimization import *
import os
import time
from sklearn.model_selection import train_test_split
from customized_packages.pyswarm import pso as pso_c
from datetime import timedelta


plt.ioff()


def plot_model(model, theta):
    """
    Gives points to plot shape of the term structure given theta and model.

    """

    model = partial(model, theta=theta)
    model = np.vectorize(model)
    x = np.linspace(0.0001, 50, 100)
    y = model(x)
    #xx = np.array([0.083333, 0.1667, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    #print(model(xx))
    return [x, y]


def bound_theta(theta, low, high):
    """
    Returns theta inside bounds if the one obtained is outside.
    Constraints are dealt with in init_points(), but bounds throw an exception

    """
    msk = theta < low
    theta[msk] = low[msk]
    msk = theta > high
    theta[msk] = high[msk]
    return theta




def on_the_run(table, now, life=1.5, age=60):
    """
    Returns a mask of assets that are not too off-the-run

    """

    age = math.floor(365*age)
    age = timedelta(days=age)
    mask = table["Life"] < life
    try:
        now = re.sub(".xlsx", "", now)
        now = datetime.strptime(now, "%y%m%d")
    except:
        pass
    mask = mask & ((now - table["Issue Date"]) < age)
    return mask


def Rsquared(P, Phat):
    """
    Calculates R2 measure when supplied with two vectors

    """

    RSS = (P - Phat) - (P - Phat).mean()
    RSS = RSS.dot(RSS.transpose())
    TSS = P - P.mean()
    TSS = TSS.dot(TSS.transpose())
    return 1 - RSS/TSS

fail = False
def get_results(algos,
                model=NSS,
                use_past_points=True,
                objective_var="Price",
                spec="NS",
                mask=[0,2],
                bounds=[np.array([0, -40, 0.001, 0, 0.001, 0, 0.001]), np.array([5, 40, 30, 40, 30, 40, 30])],
                pointsamt=100,
                miter=500,
                weight=False,
                save_st=True,
                do_plots=True,
                folder="test",
                _where=r"D:\dipl"):
    """
    A function to combine optimization and table_prep and get a table with optimization results

    """


    costfunc = {"Price": cost, "Yield": YTMcost}[objective_var]
    outdf = pd.DataFrame()
    best_theta = np.full(len(bounds[0]), np.inf)

    for day in os.listdir(_where + r"\data"):
        os.chdir(_where + r"\data")
        print(day)
        # For tests:       day = "200717.xlsx"
        tbl = clean_xl(day)
        tbl = tbl[on_the_run(tbl, day, 0.15, 1)]
        train, test = train_test_split(tbl, test_size=0.2)
        day = re.sub(".xlsx", "", day)

        if weight:
            # harmonic mean
            _correction = len(train["Duration"])/((((1/train["Duration"]).sum())/(1/train["Duration"])).sum())
        else:
            _correction = 1

        def con(theta):
            return np.array([theta[0] + theta[1]])

        low, high = bounds[0], bounds[1]
        # get D-L starting points
        if use_past_points == "DL":
            best_theta = DL_starting(train, spec)
            if (best_theta < low).any() | (best_theta > high).any():
                best_theta = bound_theta(best_theta, low, high)
        # prep objective and starting points
        obj = partial(costfunc, table=train, rmodel=model, loss=sqresid, weighted=weight)
        dst = partial(uneven_distr, where=best_theta)
        start = init_points(low, high, pointsamt, con, dst)


        if save_st or do_plots:
            try:
                os.mkdir(_where + "\\results\\" + folder)
                os.chdir(_where + "\\results\\" + folder)
            except FileExistsError:
                os.chdir(_where + "\\results\\" + folder)
            outdf.to_csv("result")
            # create folder for given day
            try:
                os.mkdir(day)
                os.chdir(_where + "\\results\\" + folder + "\\" + day)
            except FileExistsError:
                os.chdir(_where + "\\results\\" + folder + "\\" + day)

        if save_st:
            np.save("St_points", start)
            np.save("Theta_best", best_theta)

        # optimize with all supplied algorithms
        for algname, alg in algos.items():
            print(algname)
            timer = time.process_time_ns()
            res = alg(obj, low, high,
                      debug=False, f_ieqcons=con, maxiter=miter, swarmsize=pointsamt, st_points=start,
                      stopping=0, mask=mask)

            # TODO see whether is important or old version
            if (res[0] < bounds[0]).any() | (res[0] > bounds[1]).any():
                print("Out of bounds", res[0])
                global fail
                fail = True
            timer = time.process_time_ns() - timer

            if res[1] >= 1e+10: #failure to get any results
                res = [[1, 0, 0.05, 0, 0.05, np.inf], res[1]]
            timer =timer/1000000000

            # used later to construct train, test, combined results
            tbl.loc[train.axes[0], "Train"] = True
            tbl.loc[test.axes[0], "Train"] = False

            resNS = partial(model, theta=res[0])
            pest = partial(P_estimate, rfunc=resNS)
            tbl["Phat"] = tbl.apply(lambda x: pest(x["Cpn"], x["Par Amt"], x["Time Adj"]), 1)
            tbl["Yhat"] = tbl.apply(lambda x: resNS(x["Tenor"]), 1)

            # This allows to iteratively fill train, test and combined versions of results
            helperdict = {"Train": False, "Test": True, "Combined": None}
            for k, v in helperdict.items():
                temptbl = tbl[~(tbl["Train"] == v)]
                row = {"Date": day,
                       "Algorithm": algname,
                       "Data": k,
                       "Loss_YTM": partial(YTMcost, table=temptbl, rmodel=model, loss=sqresid)(res[0]),
                       "Loss_P": partial(cost, table=temptbl, rmodel=model, loss=sqresid, weighted=weight)(res[0]),
                       "R2_YTM": Rsquared(temptbl["YTM"], temptbl["Yhat"]),
                       "R2_P": Rsquared(temptbl["Midpoint"], temptbl["Phat"]),
                       "N": temptbl.shape[0],
                       "Bias": (temptbl["Phat"] - temptbl["Midpoint"]).mean(),
                       "Time": timer,
                       "Theta": res[0],
                       "Tenors": train["Tenor"].to_numpy(),
                       "RMSE_P": RMSE(temptbl["Midpoint"], temptbl["Phat"]),
                       "RMSE_YTM": RMSE(temptbl["Yhat"], temptbl["YTM"]),
                       "RMSE_P_S": RMSE(temptbl.loc[temptbl["Tenor"] <= 3, "Midpoint"],
                                      temptbl.loc[temptbl["Tenor"] <= 3, "Phat"]),
                       "RMSE_P_M": RMSE(temptbl.loc[(temptbl["Tenor"] > 3) & (temptbl["Tenor"] <= 10), "Midpoint"],
                                        temptbl.loc[(temptbl["Tenor"] > 3) & (temptbl["Tenor"] <= 10), "Phat"]),
                       "RMSE_P_L": RMSE(temptbl.loc[temptbl["Tenor"] > 10, "Midpoint"],
                                        temptbl.loc[temptbl["Tenor"] > 10, "Phat"]),
                       "RMSE_YTM_S": RMSE(temptbl.loc[temptbl["Tenor"] <= 3, "YTM"],
                                        temptbl.loc[temptbl["Tenor"] <= 3, "Yhat"]),
                       "RMSE_YTM_M": RMSE(temptbl.loc[(temptbl["Tenor"] > 3) & (temptbl["Tenor"] <= 10), "YTM"],
                                        temptbl.loc[(temptbl["Tenor"] > 3) & (temptbl["Tenor"] <= 10), "Yhat"]),
                       "RMSE_YTM_L": RMSE(temptbl.loc[temptbl["Tenor"] > 10, "YTM"],
                                        temptbl.loc[temptbl["Tenor"] > 10, "Yhat"])
                       }
                print(row["Loss_P"], row["N"], row["Time"])
                outdf = outdf.append(row, ignore_index=True)

            # can plot results, but will take a lot of space. Good to use on small runs
            if do_plots:
                fig, axs = plt.subplots(1, 2)
                colours = {True: 'green', False: 'red'}
                axs[0].scatter(tbl["Phat"], tbl["Midpoint"], c=tbl["Train"].map(colours), s=2)
                axs[0].axline([0, 0], [1, 1])
                axs[0].axis(ymin=min(tbl["Midpoint"]) - 2, ymax=max(tbl["Midpoint"]) + 2,
                            xmin=min(tbl["Phat"]) - 2, xmax=max(tbl["Phat"]) + 2)
                t = plot_model(NSS, res[0])
                axs[1].plot(t[0], t[1], "r")
                axs[1].scatter(tbl["Tenor"], tbl["YTM"], c=tbl["Train"].map(colours), s=2)
                axs[1].axis(ymin=0, ymax=max(tbl["YTM"]) + 0.01,
                            xmin=0, xmax=32)

                fig.savefig(algname)

        # get best theta of the run to generate next starting points around
        if (use_past_points is True) & (use_past_points != "DL"):
            _lossdict = {"Price": "Loss_P", "Yield": "Loss_YTM"}
            temp = outdf.loc[(outdf["Date"] == day) & (outdf["Data"] == "Combined"), :]
            temp = temp.sort_values(_lossdict[objective_var], ascending=True)
            temp = temp.drop_duplicates("Date")
            print(temp)
            best_theta = temp["Theta"].item()
            if (best_theta < low).any() | (best_theta > high).any():
                best_theta = bound_theta(best_theta, low, high)
            print(best_theta)

    return outdf


# dictionary to input algos used
algdict = {"Nelder-Mead": nelder_mead, "Diff_evo": diff_evo, "COBYLA": COBYLA,
           "Dual_annealing": d_annealing, "L-BFGS-B": LBFGSB}

a = get_results(algdict,
                model=NSS_red,
                use_past_points="DL",
                spec="NSS_red",
                objective_var="Price",
                bounds=[np.array([0, -5, 0.001, -20]), np.array([10, 35, 35, 35])],
                mask=[0,2],
                pointsamt=400,
                miter=200,
                save_st=True,
                weight=False,
                do_plots=False,
                folder="DL_points_price_NSS",
                _where=r"D:\dipl")
a.to_csv(r"D:\dipl\results" + "\\DL_points_price_NSS\\" + "result_fin")
# *_points_price_NSS good1

a = get_results(algdict,
                model=NSS_red,
                use_past_points=False,
                spec="NSS_red",
                objective_var="Price",
                bounds=[np.array([0, -5, 0.001, -20]), np.array([10, 35, 35, 35])],
                mask=[0,2],
                pointsamt=400,
                miter=200,
                weight=True,
                save_st=True,
                do_plots=False,
                folder="random_points_price_NSS_w_free",
                _where=r"D:\dipl")
a.to_csv(r"D:\dipl\results" + "\\random_points_price_NS_w\\" + "result_fin")
# *_points_price_NSS_w good2

strtp = np.load(r"D:\dipl\results\past_points_price\201127\St_points.npy")















