# -*- coding: utf-8 -*-

import math

import numpy as np
import pyswarm as pso
from functools import partial
from scipy.stats import truncnorm
import scipy as sp
from customized_packages.diff_evo import _differentialevolution as de
from sklearn import linear_model
import time
from customized_packages.dual_annealing import _dual_annealing as da


class breaker:
    """
    Class used to record result and time of threshold optimization while the full one is executed.
    start timer on init, each iteration compare result with threshold first time it is lower record time and the result
    """

    def __init__(self, stopper):
        self.timer = time.process_time_ns()
        self.stopper = stopper
        self.status = True
        self.results = [np.inf, np.inf]

    def check_stopping(self, val, x):
        if (val <= self.stopper) & self.status:
            self.status = False
            self.results = [x, (time.process_time_ns() - self.timer)/1e+9]


def NSS_red(t, theta):
    """
    NSS model with restrictions against multicollinearity

    """
    tau = theta[2] + 1
    beta = theta[3] + 1
    return theta[0] + theta[1] * ((1 - math.exp(-t / theta[2])) / (t / theta[2])) + \
           theta[3] * ((1 - math.exp(-t / theta[2])) / (t / theta[2]) - math.exp(-t / theta[2])) + \
           beta * ((1 - math.exp(-t / tau)) / (t / tau) - math.exp(-t / tau))

def NS_red(t, theta):
    """
    NS model with restrictions against multicollinearity

    """
    tau = theta[2] + 1
    return theta[0] + theta[1] * ((1 - math.exp(-t / theta[2])) / (t / theta[2])) + \
           theta[3] * ((1 - math.exp(-t / tau)) / (t / tau) - math.exp(-t / tau))


def NSS(t, theta):
    """
    Pure NSS model

    """
    return theta[0] + theta[1]*((1-math.exp(-t/theta[2]))/(t/theta[2])) + \
           theta[3]*((1-math.exp(-t/theta[4]))/(t/theta[4])-math.exp(-t/theta[4])) + \
           theta[5]*((1-math.exp(-t/theta[6]))/(t/theta[6])-math.exp(-t/theta[6]))


def NS(t, theta):
    """
    Pure NS model

    """
    return theta[0] + theta[1]*((1-math.exp(-t/theta[2]))/(t/theta[2])) + \
           theta[3]*((1-math.exp(-t/theta[4]))/(t/theta[4])-math.exp(-t/theta[4]))


def P_estimate(coupon, par, tadj, rfunc):
    """
    Takes coupon payment as a single number, par value, time to payments, and term structure function to return
    estimated price
    rfunc: R^1 -> R^1

    """
    # rfunc is NSS with theta plugged in lambda x: NSS(x, theta)
    Phat = 0
    for t in tadj:
        Phat += coupon*math.exp(-t*rfunc(t))
    Phat += par*math.exp(-tadj[-1]*rfunc(tadj[-1]))
    return Phat


def wrap_model(row, model):
    """
    A small helper function to use with lambda in apply

    """
    return P_estimate(row["Cpn"], row["Par Amt"], row["Time Adj"], model)


def sqresid(P, Phat, w=1):
    """
    Given two vectors returns RSS or 1.797e+300 if overflow happened

    """
    try:
        return sum(w*(P - Phat)**2)
    except OverflowError:
        return 1.797e+300




def cost(theta, table, rmodel, loss, weighted=False):
    """
    Takes theta vector, table to apply equatons to, term structure function and loss function to get loss of the
    whole dataset. Can be done with invesrse duration weitghing.
    Returns loss value

    """

    rmodel = partial(rmodel, theta=theta)
    try:
        Phat = table.apply(lambda x: wrap_model(x, model=rmodel), 1)
    except OverflowError:
        Phat = 1.797e+100
    P = table["Midpoint"]
    if weighted:
        return loss(P, Phat, (1/table["Duration"])/((1/table["Duration"]).sum()))
    return loss(P, Phat)

# ss = lambda theta: cost(test, NSS, sqersid, theta)
# pso(ss)


def RMSE(P, Phat):
    """
    Takes two vectors, returns their RMSE or np.nan if they are empty

    """
    if P.empty:
        return np.nan
    return math.sqrt((P - Phat).T.dot(P - Phat)/len(P))


def DL_starting(table, spec):
    """
    Supply table - get parameters similar to Diebold-Li procedure, without bootstrapping or zero-coupon bonds though.
    spec allows for different model specifications. Now unrestriced NS is implemented (spec=None) and
    restricted NSS (spec=NSS_red)

    """
    # loading 1
    X = table.apply(lambda t: (1-math.exp(-t["Tenor"]*0.0609))/(t["Tenor"]*0.0609), 1).to_numpy()
    # loading 2
    temp = table.apply(lambda t: (1-math.exp(-t["Tenor"]*0.0609))/(t["Tenor"]*0.0609)-math.exp(-t["Tenor"]*0.0609), 1).to_numpy()
    X = np.append([X], [temp], axis=0).transpose()
    # run regression and save coefficients
    reg = linear_model.LinearRegression()
    reg.fit(X, table["YTM"].to_numpy().reshape(-1, 1))
    thetas = np.array([reg.intercept_[0], reg.coef_[0][0], 1/0.0609, reg.coef_[0][1], 1/0.0609])
    if spec == "NSS_red":
        thetas = thetas[:-1]
    return thetas

"""

reg = DL_starting(tbl)
tbl["Pred"] = tbl.apply(lambda t: reg.intercept_ + reg.coef_[0][0]*(1-math.exp(-t["Tenor"]*0.0609))/(t["Tenor"]*0.0609) + reg.coef_[0][1]*((1-math.exp(-t["Tenor"]*0.0609))/(t["Tenor"]*0.0609)-math.exp(-t["Tenor"]*0.0609)), 1)
fig, ax = plt.subplots()
ax.scatter(tbl["Tenor"], tbl["YTM"])
ax.scatter(tbl["Tenor"], tbl["Pred"])
fig.show()
"""

def init_points(lb, ub, size, cnstrt=None, distr=None):
    """
    Supply lower bound, upper bound, amount of points to be generated constraint as a function and possibly a
    distribution - get starting points: 1 row - one point
    if distr is None get uniform one

    """

    if cnstrt is None:
        cnstrt = lambda x: 5
    # generate initial values
    dim = len(ub)
    # make callable distr function to use later on
    if distr is None:
        distr = lambda a: np.random.rand(a, dim)
    else:
        distr = partial(distr, lb=lb, ub=ub, scale=1)
    x = distr(size)
    x = np.tile(lb, (size, 1)) + np.multiply(x, np.tile((ub-lb), (size, 1)))
    # correct the ones not satisfying constraints excluding the first one (best guess) I believe was not necessary,
    # but still done to not get stuck in an endless while
    for i in range(1, size):
        while np.all(cnstrt(x[i, ]) < 0):
            x[i, ] = distr(1)
            x[i, ] = lb + x[i, ] * (ub - lb)
    return x


def uneven_distr(size, where, lb, ub, scale=1):
    """
    Supply amount of points to be generated, best guess, lower and upper bounds and scale rescaling parameter
    (see scalemod variable) - get truncated normal distribution of points. Where np.inf are placed in vectors
    uniform distribution is used. All np.inf will result in uniform distr.

    """
    # np.inf where no info should be supplied to generate uniform variables
    where_unit = (where-lb)/(ub-lb)
    result = np.zeros([size, len(where)])
    for i in range(len(where_unit)):
        temp = where_unit[i]
        if np.isinf(temp):
            result[:, i] = np.random.rand(size)
        else:
            # arbitrary decision as to how to set scale depending on position: 0.4/([0.5 - |0.5-pos|] - 0.5)
            # seemed to do well (closer to bound - has to set scale higher to get more points on the opposite bound)
            # can rescale the scale. have to do some more work to change functional dependence
            scalemod = (0.4/(0.5 - abs(temp - 0.5) + 0.5))*scale
            result[:, i] = truncnorm.rvs((-temp) / scalemod, (1 - temp) / scalemod, temp, scalemod, size)
    result[0, ~np.isinf(where_unit)] = where_unit[~np.isinf(where_unit)]
    return result

# !!!!!!!!!!!!!!
""" All optimizers defined here take same inputs. objective function, lower, upper bounds, debug parameter,
    constraints, maximum number of iterations, swarmsize, starting points, stopping value of obj, and 
    list of indices of the important constraints (beta0 and taus)
    
    obj - the function to minimize, already should be of 1 argument
    low, high - lower and higher bounds respectively
    debug - debug argument for algorithms that have one 
    f_ieqcons - constraints that are in the form c(theta) >= 0
    maxiter - amount of iterations before stopping
    swarmsize - number of points used in optimization
    st_points - starting points generated in advance
    strat - only used for differetial evolution
    stopping - threshold to save the fast optimization result
    mask - important constraints
    
    Adjusted for different algorithms inside their respective functions e.g. dual annealing will use 14*maxiter
    results are [argmin(1) obj(), min(1) obj(), argmin(2) obj(), time(2)] where first two relate to full optimization
    and the other two to threshold one
    """
# !!!!!!!!!!!!!!
def diff_evo(obj, low, high, debug, f_ieqcons, maxiter, swarmsize, st_points, strat="best1bin", stopping=0,
             mask=[0,2]):

    bounds = [(low[i], high[i]) for i in range(len(low))]
    constr = sp.optimize.NonlinearConstraint(f_ieqcons, 0, np.inf)
    res = de.differential_evolution(obj, bounds, strategy=strat, maxiter=math.ceil(0.8*maxiter), popsize=swarmsize,
                                    disp=debug, init=st_points, constraints=(constr), stopping=stopping)
    res = [res["x"], res["fun"], res["xrestr"], res["timerestr"]]
    return res



def wrap_const(val, low, high, obj, con, mask=[0, 2], _da=False):
    """
    Change objective function to punish for important constraint violation.
    Takes theta, upper, lower bounds, objective function, constraints and mask as the one before. Returns merit function
    with constraints incorporated.

    """

    penalty = 0
    valm, lowm, highm = val[mask], low[mask], high[mask]
    if ((valm < lowm).any()):
        #    return 1e+300
        ###
        msk = valm < lowm
        penalty += sum(lowm[msk] - valm[msk])
    elif con(val) < 0:
        penalty += -con(val)
        if da:
            penalty = penalty + 1000
    return obj(val) + penalty


def check_domain(val, low, high, cons, mask=[0, 2]):
    """
    Check all important bounds/constraints to save any given result in iterative optimizations.

    """
    return (val[mask] >= low[mask]).all() & (cons(val) >= 0)


def nelder_mead(obj, low, high, debug, f_ieqcons, maxiter, swarmsize, st_points, stopping, mask=[0,2]):

    obj = partial(wrap_const, low=low, high=high, obj=obj, con=f_ieqcons, mask=mask)
    st_points = st_points[np.random.choice(st_points.shape[0], math.ceil(st_points.shape[0]/len(low)),
                                           replace=False), :]

    res = [0, 1e+10, np.inf, np.inf]
    altres = breaker(stopping)
    for st in st_points:
        #simplex = generate_simplex(st, low, high)
        sol = sp.optimize.minimize(obj, st, method="Nelder-Mead", options={"disp": debug, "maxiter": maxiter})
        if (sol["fun"] < res[1]) & check_domain(sol["x"], low, high, f_ieqcons, mask=mask):
            res[0], res[1] = [sol["x"], sol["fun"]]
        altres.check_stopping(sol["fun"], sol["x"])
        res[2], res[3] = altres.results
    return res



def d_annealing(obj, low, high, debug, f_ieqcons, maxiter, swarmsize, st_points, stopping, mask=[0,2]):

    maxiter = int(maxiter*14)
    obj = partial(wrap_const, low=low, high=high, obj=obj, con=f_ieqcons, mask=mask, _da=True)
    st_points = st_points[0, :]
    sol = da.dual_annealing(obj, bounds=list(zip(low, high)), maxiter=maxiter, x0=st_points,
                                     stopping=stopping)
    res = [sol["x"], sol["fun"], sol["xrestr"], sol["timerestr"]]
    return res


def COBYLA(obj, low, high, debug, f_ieqcons, maxiter, swarmsize, st_points, stopping, mask=[0,2]):

    #obj = partial(wrap_const, low=low, high=high, obj=obj, con=f_ieqcons)
    constr1 = sp.optimize.NonlinearConstraint(f_ieqcons, 0+0.0002, np.inf)
    constr2 = sp.optimize.NonlinearConstraint(lambda x: x[mask], low[mask] + 0.0002, np.inf)
    st_points = st_points[np.random.choice(st_points.shape[0], math.ceil(0.66*st_points.shape[0]),
                                           replace=False), :]
    result = [0, 1e+10, np.inf, np.inf]
    altres = breaker(stopping)
    for st in st_points:
        sol = sp.optimize.minimize(obj, st,  constraints=(constr1, constr2), method="COBYLA",
                                   options={"disp": debug, "maxiter": maxiter})
        if (sol["fun"] < result[1]) & check_domain(sol["x"], low, high, f_ieqcons, mask=mask):
            result[0], result[1] = [sol["x"], sol["fun"]]
        altres.check_stopping(sol["fun"], sol["x"])
        result[2], result[3] = altres.results
    return result


def LBFGSB(obj, low, high, debug, f_ieqcons, maxiter, swarmsize, st_points, stopping, mask=[0,2]):

    obj = partial(wrap_const, low=low, high=high, obj=obj, con=f_ieqcons, mask=mask)
    st_points = st_points[
                np.random.choice(st_points.shape[0], math.ceil(st_points.shape[0] / 2.5), replace=False), :]
    bounds = [(low[i], high[i]) for i in range(len(low))]
    result = [0, 1e+10, np.inf, np.inf]
    altres = breaker(stopping)
    for st in st_points:
        sol = sp.optimize.minimize(obj, st, bounds=bounds, method="L-BFGS-B", options={"maxcor": 5, "disp": debug, "maxiter": maxiter})
        if (sol["fun"] < result[1]) & check_domain(sol["x"], low, high, f_ieqcons, mask=mask):
            result[0], result[1] = [sol["x"], sol["fun"]]
        altres.check_stopping(sol["fun"], sol["x"])
        result[2], result[3] = altres.results
    return result






def YTMcost(theta, table, rmodel, loss):
    """
    Not used but should work:
    Same as cost but with YTM rather than price + no weighting and loss is multiplied by 1000 for readability

    """

    rmodel = partial(rmodel, theta=theta)
    try:
        Yhat = table.apply(lambda x: rmodel(x["Tenor"]), 1)
    except OverflowError:
        Yhat = 1.797e+100
    Y = table["YTM"]
    return 1000*loss(Y, Yhat)


def generate_simplex(st, low, high, _scale=0.1):
    """
    Not used. provided a best guess, bounds and scale generates a simplex to start with.
    Tried and in the runs where default simplex generation did not work this did not either.

    """
    dim = len(st)
    simplex = np.zeros((dim+1, dim))
    simplex[0, :] = st
    for i in range(0, dim):
        dir = np.zeros(dim)
        dir[i] = 1
        simplex[i+1, :] = st + dir*_scale*(high-low)
    return simplex


if __name__ == "__main__":
    from table_prep import *
    import os
    os.chdir(r"D:\dipl\data")
    tryon = "201002.xlsx"
    test = clean_xl(tryon)
    def con(theta):
        return [theta[0] + theta[1]]
    testt = test.iloc[1:2, ]
    obj = partial(cost, table=testt, rmodel=NS, loss=sqresid)
    res = pso.pso(obj, [0, -20, 0, -20, 0], [20, 20, 30, 30, 30], debug=True, maxiter=100)
    #obj([1, 2, 5, 6, 2])
    resNS = partial(NS, theta=res[0])
    pest = partial(P_estimate, rfunc=resNS)
    test["Phat"] = test.apply(lambda x: pest(x["Cpn"], x["Par Amt"], x["Time Adj"]), 1)
    cost(res[0], test, NS, sqresid)
    import matplotlib.pyplot as plt
    plt.scatter(test["Phat"], test["Ask Price"])
    plt.axline([0, 0], [1, 1])
    plt.show()


#[6.11798901, -6.05839892, 6.12378188, -14.54613799,15.01909399]

    points = uneven_distr(np.array([19, np.inf]), np.array([0, -20]), np.array([20, 20]), 500)
    points = np.tile(np.array([0, -20]), (500, 1)) + np.multiply(points, np.tile((np.array([20, 20]) -
                                                                                   np.array([0, -20])), (500, 1)))
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()
    plt.hist(points[:, 0])
    points[:, 0].mean()
    # 10 0.4, 5 0.5 2.5 0.6 1 0.7 0.5 0.75 0.25 0.8


    def con(theta):
        return np.array([theta[0] + theta[1]])
    dst = partial(uneven_distr, where=np.array([4, np.inf, np.inf, -10, np.inf]))
    ttt = init_points(np.array([0, -20, 0, -20, 0]), np.array([20, 20, 30, 30, 30]), 1000, con, dst)
    plt.hist(ttt[:, 1])
    # ^^ constraints change theta[1] to non-uniform one without them all looks as expected

    penalty = 0
    if ((val < low).any()) | ((val > high).any()):
        #    return 1e+300
        ###
        msk = val < low
        penalty += sum(low[msk] - val[msk])
        msk = val > high
        penalty += sum(val[msk] - high[msk])
    elif con(val) < 0:
        penalty += -con(val)
    #return (2 ** (penalty + 1)) * obj(val)
    def wrap_const(val, low, high, obj, con):

        penalty = 0
        if ((val < low).any()) | ((val > high).any()):
            #    return 1e+300
            ###
            msk = val < low
            penalty += sum(low[msk] - val[msk])
            msk = val > high
            penalty += sum(val[msk] - high[msk])
        elif con(val) < 0:
            penalty += -con(val)

        return obj(val) + penalty

