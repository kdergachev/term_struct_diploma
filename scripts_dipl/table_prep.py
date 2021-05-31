# -*- coding: utf-8 -*-
import math

import pandas as pd
import os
from datetime import datetime
import re
import numpy as np
from dateutil.relativedelta import relativedelta
import QuantLib as ql
import scipy.optimize as sopt
from functools import partial


def get_dates(folder=os.getcwd()):
    """
    Input path to a folder with .xlsx files only with names in "%y%m%d" format, returns list of datetime

    """
    
    
    res = []
    files = os.listdir(folder)
    for i in files:
        i = re.sub(".xlsx", "", i)
        i = datetime.strptime(i, "%y%m%d")
        res.append(i)
    return res


# helper functions to hack date range functionality together

def to_qldate(x):
    """
    
    Turn other date formats to ql.Date one
    
    """
    return ql.Date(x.day, x.month, x.year)


def adjustment(cpns, cal=ql.UnitedStates(ql.UnitedStates.FederalReserve), 
               conv=ql.Following):
    """
    
    To be used in list comprehension to apply following convention to a schedule
    
    """
    return cal.adjust(cpns, conv)


def populate_coupons(row, nowdate):
    """
    given a row and date the data was collected returns a list of time in years to each of the remaining coupon payments
    used in apply as well
    """


    if row["Ticker"] == "T":
        if to_qldate(nowdate) >= to_qldate(row["Maturity"]):
            return np.nan
        else:
            coupons = ql.MakeSchedule(to_qldate(nowdate),
                                      to_qldate(row["Maturity"]),
                                      ql.Period('6M'),
                                      backwards=True,
                                      endOfMonth=True)

        coupons = [adjustment(cpns=i) for i in coupons]
        coupons = [ql.ActualActual().yearFraction(coupons[0],coupons[i]) for i in range(1, len(coupons))]
    else:
        coupons = ql.ActualActual().yearFraction(to_qldate(nowdate),
                                                 to_qldate(row["Maturity"])) # list unchecked
    return coupons


def pv(rate, cpn, par, tadj):
    """
    given a rate , coupon amount, par value and list of when the coupons are paid returns present value of said bond
    rate can be both constant for YTM and a function of time for NS, NSS... runs
    """
    if not hasattr(rate, "__call__"):
        r = lambda x: rate
    else:
        r = rate
    Phat = 0
    for t in tadj:
        Phat += cpn*math.exp(-t*r(t))
    Phat += par * math.exp(-tadj[-1] * r(tadj[-1]))
    return Phat


def ytm(cpn, par, tadj, price):
    """
    given coupon payment, par value, list of time to payment and price of the bond returns yield to maturity
    again used in apply to get YTM
    """
    pvpart = partial(pv, cpn=cpn, par=par, tadj=tadj)
    npv = lambda x: price - pvpart(x)
    return sopt.root(npv, 0)["x"]


def dur(row):
    """
    Given a row returns Macaulay duration
    used in apply
    """
    D = 0
    for i in row["Time Adj"]:
        D += i*row["Cpn"]*math.exp(-i*row["YTM"])
    D += i*100*math.exp(-i*row["YTM"])
    return D/row["Midpoint"]
    
def clean_xl(filename):
    """
    Does all together for a given .xlsx file:
    loads, gets treasury instruments, drops NA, semiannualizes the coupons, adds Life column, times to payments, gets
    prices of bills in dollars, midpoint, YTM, Duration, time to maturity (called Tenor)
    """
    
    # get date data was taken
    nowdate = re.sub(".xlsx", "", filename)
    nowdate = datetime.strptime(nowdate, "%y%m%d")
    names = ["United States Treasury Note/Bond", "United States Treasury Bill"]
    # load and clean
    table = pd.read_excel(filename, na_values=['--'])
    table = table.dropna(1, how="all")
    table = table[table["Issuer Name"].isin(names)]
    table["Maturity"] = pd.to_datetime(table["Maturity"])
    table["Issue Date"] = pd.to_datetime(table["Issue Date"])
    # semiannualise coupons
    table["Cpn"] = table["Cpn"]/2

    # deals with Par Amt, Minimum Increment uncertainty
    try:
        _try = table["Par Amt"]
    except:
        table["Par Amt"] = table["Minimum Increment"]
        #table["Par Amt"] = 100


    # add how much of security's life has passed
    table["Life"] = (nowdate - table["Issue Date"]) / (table["Maturity"] - table["Issue Date"])
    # get time adjustments for discounting
    table["Time Adj"] = table.apply(lambda x: populate_coupons(x, nowdate), 1)
    table = table.loc[~table["Time Adj"].isna(), :]
    # here get prices of bills to use in models
    table_mask = table["Ticker"] == "B"
    nowdate = to_qldate(nowdate)
    if nowdate.isLeap(nowdate.year()):
        adj = 366/360
    else:
        adj = 365/360
    table.loc[table_mask, "Ask Price"] = table.loc[table_mask, "Par Amt"]*(1 - (table.loc[table_mask, "Ask Price"]/100)
                                                                           * table.loc[table_mask, "Time Adj"] * adj)
    table.loc[table_mask, "Bid Price"] = table.loc[table_mask, "Par Amt"] * (
                1 - (table.loc[table_mask, "Bid Price"] / 100)
                * table.loc[table_mask, "Time Adj"] * adj)
    # put tenors for bills in lists to use in functions later
    table.loc[table_mask, "Time Adj"] = table.loc[table_mask, "Time Adj"].apply(lambda x: [x], 1)
    # add midpoint column
    table["Midpoint"] = (table["Bid Price"] + table["Ask Price"])/2
    # add YTM
    table["YTM"] = table.apply(lambda x: ytm(x["Cpn"], x["Par Amt"], x["Time Adj"], x["Midpoint"]).item(), 1)
    # add maturity column
    table["Tenor"] = table.apply(lambda x: x["Time Adj"][-1], 1)
    table["Duration"] = table.apply(lambda x: dur(x), 1)
    return table


if __name__ == "__main__":
    os.chdir(r"D:\dipl\data")
    tryon = "190607.xlsx"
    #print(get_dates())
    test = clean_xl(tryon)
    tryon = re.sub(".xlsx", "", tryon)
    print(populate_coupons(test.iloc[6, ], datetime.strptime(tryon, "%y%m%d")))
    sss = datetime.strptime(tryon, "%y%m%d")
    populate_coupons(test.iloc[71, ], sss)

    # semi annual test
    for i in os.listdir():
        print(i)
        test = clean_xl(i)
        if (test["Par Amt"] == 100).any():
            pass
        else:
            print(i, "wrong")






