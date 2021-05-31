import math
from table_prep import *
import matplotlib.pyplot as plt
from optimization import *
import datetime


datadir = r"D:\dipl\data"
os.chdir(r"D:\dipl\data")
tryon = "181220.xlsx"
print(get_dates(datadir))
test = clean_xl(tryon)
plt.scatter(test["Tenor"], test["YTM"])
plt.show()
minusshortbond = test[test["Ticker"] == "B"]
plt.scatter(minusshortbond["Tenor"], minusshortbond["YTM"])
plt.show()
# see why YTM is so wrong











tryon = "181220.xlsx"
testt = clean_xl(tryon)
#testt = test#.iloc[0:100, ]
#tryon = re.sub(".xlsx", "", tryon)
#tryon = datetime.datetime.strptime(tryon, "%d%m%y")
#testt["Life"] = (tryon - testt["Issue Date"])/(testt["Maturity"] - testt["Issue Date"])
testt = testt[(testt["Life"] > 0.10) & (testt["Life"] < 0.30)]
plt.scatter(testt["Tenor"], testt["YTM"])
plt.show()

def con(theta):
    return [theta[0] + theta[1]]
obj = partial(cost, table=testt, rmodel=NSS, loss=sqresid)
res = pso.pso(obj, [0, -40, 0.000001, -40, 0.000001, -40, 0.000001], [5, 40, 30, 40, 30, 40, 30],
              debug=True, f_ieqcons=con, maxiter=100, swarmsize=500)

resNS = partial(NSS, theta=res[0])
pest = partial(P_estimate, rfunc=resNS)
testt["Phat"] = testt.apply(lambda x: pest(x["Cpn"], x["Par Amt"], x["Time Adj"]), 1)

fig, axs = plt.subplots(1, 2)
axs[0].scatter(testt["Phat"], testt["Midpoint"], s=2)
axs[0].axline([0, 0], [1, 1])


def plot_model(model, theta):
    model = partial(model, theta=theta)
    model = np.vectorize(model)
    x = np.linspace(0.0001, 50, 100)
    y = model(x)
    out = plt.plot(x, y, 'r')
    xx = np.array([0.083333, 0.1667, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    print(model(xx))
    return [x, y]

t = plot_model(NSS, res[0])
axs[1].plot(t[0], t[1], "r")


tsttheta = [1.48141024, -1.48140607, 24.59498243,  4.26354728, 21.03515285, -6.85932207, 28.55870205]
nss = partial(NSS, theta=tsttheta)
testt["Phat"] = testt.apply(lambda x: P_estimate(x["Cpn"], x["Par Amt"], x["Time Adj"], nss), 1)

plt.scatter(testt["Phat"], testt["Midpoint"], s=2)
plt.axline([0, 0], [1, 1])
plt.show()






import math
from table_prep import *
import matplotlib.pyplot as plt
from optimization import *


datadir = r"D:\dipl\data"
os.chdir(r"D:\dipl\data")
for day in os.listdir(datadir):
    tbl = clean_xl(day)










timer = time.process_time_ns()
for j in range(100):
    for i in range(10000):
        r = math.exp(-i* 0.05)
    timer = time.process_time_ns() - timer

timer =timer/1000000000

use_past_points = "DL"
if (use_past_points is True) & (use_past_points != "DL"):
    print("aaaaaa")




def get_tabular_datatwo(table, what, _over="Algorithm"):

    # whatlist.append(_over)
    over = ["Train", "Test"]
    table = table.loc[:, ["Data"] + [what] + [_over]]
    outtb = pd.DataFrame()
    for i in table[_over].unique():
        row = {"min": [[1,1]], "0.25": [[1,1]], "median": [[1,1]], "0.75": [[1,1]], "max": [[1,1]],
               "mean": [[1,1]], "stdev": [[1,1]]}
        for j, k in enumerate(over):
            temp = table.loc[(table["Data"] == over[j]) & (table[_over] == i), :]
            print(j)
            row["min"][0][j] = round(temp[what].min(), 2)
            row["0.25"][0][j] = round(temp[what].quantile(0.25), 2)
            row["median"][0][j] = round(temp[what].quantile(0.5), 2)
            row["0.75"][0][j] = round(temp[what].quantile(0.75), 2)
            row["max"][0][j] = round(temp[what].max(), 2)
            row["mean"][0][j] = round(temp[what].mean(), 2)
            row["stdev"][0][j] = round(math.sqrt(temp[what].var()), 2)
            print(row)
        row = pd.DataFrame(row, index=[i])
        print(row)
        outtb = outtb.append(row)
    return outtb


# TODO finish function






class breaker:

    def __init__(self):
        self.counter = 0

    def increment(self, limit):
        self.counter += 1
        if self.counter >= limit:
            return False
        else:
            return True


tt = breaker()
for i in range(5, 15):
    print(i)
    if not tt.increment(0):
        break





