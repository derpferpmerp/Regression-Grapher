import json
import math
import random
import re
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
from cgrid import main as gcoords


subprocess.call("del tmp.json", shell=True)
subprocess.call("del hello.png", shell=True)
plt.style.use("dark_background")
sg.theme("Black")
args = sys.argv[1::] if len(sys.argv) > 1 else None

METHODS = ["log", "lin", "quad", "sig", "all"]


def calc_avg(l):
    return sum(l) / len(l)


def remove_trailing(num):
    return np.format_float_positional(num, trim="-")


def calc_sxx_or_sxy(l):
    avg = calc_avg(l)
    sxxval = 0
    for i in range(len(l)):
        sxxval += math.pow((l[i] - avg), 2)
    return sxxval


def calc_sxy(l_x, l_y):
    avg_x, avg_y = [calc_avg(l_x), calc_avg(l_y)]
    sxyval = 0
    if len(l_x) != len(l_y):
        return "Unequal Set of Lists"
    for i in range(len(l_x)):
        sxyval += (l_x[i] - avg_x) * (l_y[i] - avg_y)
    return sxyval


def gen_eq(yint, slp, mth):
    mth = mth.lower()
    if mth == "log":
        return f"y = {yint} + {slp}ln(x)"
    elif mth == "lin":
        return f"y = {yint} + {slp}x"


def log_regress(xl, yl, rreturn=False, graphing=False):
    xafterln = []
    for beforeln_x in xl:
        try:
            if beforeln_x <= 0:
                sys.exit('\nUnexpected Value "0" Found for Logarithmic Regression\n')
            logged = np.log(beforeln_x)
            xafterln.append(logged)
        except RuntimeWarning:
            return None
    SXX = calc_sxx_or_sxy(xafterln)
    SXY = calc_sxy(xafterln, yl)
    calc_sxx_or_sxy(yl)
    B2 = SXY / SXX
    B1 = calc_avg(yl) - (B2 * calc_avg(xafterln))
    if rreturn:
        original = yl
        predicted = [[B1 + (B2 * np.log(x))][0] for x in yl]
        return [original, predicted]
    elif graphing:
        original = np.linspace(min(xlist), max(xlist), max(ylist))
        expd = np.array(list(map(np.log, original)))
        expd2 = expd * B2
        expd3 = expd2 + B1
        predicted = expd3
        return [original, predicted]
    else:
        return gen_eq(B1, B2, "log")


def lin_regress(x1, y1, rreturn=False, graphing=False):
    slope = calc_sxy(x1, y1) / calc_sxx_or_sxy(x1)
    intercept = calc_avg(y1) - (slope * calc_avg(x1))
    if rreturn:
        original = y1
        predicted = [[intercept + (slope * x)][0] for x in y1]
        return [original, predicted]
    elif graphing:
        original = np.linspace(min(xlist), max(xlist), max(ylist))
        predicted_part1 = original * slope
        predicted = predicted_part1 + intercept
        return [original, predicted]
    else:
        return gen_eq(intercept, slope, "lin")


def quad_regress(xl, yl, rreturn=False, graphing=False):
    x1, x2, x3 = xl[0:3]
    y1, y2, y3 = yl[0:3]
    bottom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    try:
        a_eq = ((x3 * (y2 - y1)) + (x2 * (y1 - y3)) + (x1 * (y3 - y2))) / bottom
        b_eq = (
            ((x3 * x3) * (y1 - y2)) + ((x1 * x1) * (y2 - y3)) + ((x2 * x2) * (y3 - y1))
        ) / bottom
        c_eq = (
            (x3) * (x2 * (x2 - x3) * y1 + (x1) * (x3 - x1) * y2)
            + (x1) * (x1 - x2) * (x2 * y3)
        ) / bottom
    except ZeroDivisionError:
        return None
    a_eq, b_eq, c_eq = (remove_trailing(n) for n in [a_eq, b_eq, c_eq])
    c_eq_str = str(c_eq)
    af, bf, cf = [float(a_eq), float(b_eq), float(c_eq)]
    if cf >= 0:
        eq = f"y = {a_eq}x^2 + {b_eq}x + {c_eq}"
    else:
        eq = f"y = {a_eq}x^2 + {b_eq}x - {c_eq_str[1::]}"
    if rreturn:
        original = yl
        predicted = [[(af * (x ** 2)) + (bf * x) + cf][0] for x in yl]
        return [original, predicted]
    elif graphing:
        original = np.linspace(min(xlist), max(xlist), max(ylist))
        predicted = (af * (original ** 2)) + (bf * original) + cf
        return [original, predicted]
    else:
        return eq


def sm(lb, ub, expres, fval=2, fc=False):
    outval = 0
    expres = re.sub(r"@", "{}", expres)
    fval = len(re.findall(r"{}", expres))
    for x in range(lb, ub + 1):
        flist = []
        for g in range(fval):
            flist.append(x)
        outval += eval(expres.format(*flist))
    return outval


def norm(lstin):
    lstout = []
    maxnum = max(lstin)
    lstin2 = [itm for itm in lstin if itm != maxnum]
    lstin = lstin2
    for x in lstin:
        lstout.append(x / maxnum)
    lstout.append(1)
    return lstout


def calc_diff(original, predicted):
    lstout = []
    for l in range(len(original)):
        try:
            lstout.append(original[l] - predicted[l])
        except IndexError:
            continue
    return lstout


def calc_mse(original, predicted):
    diff = calc_diff(original, predicted)
    return round(sum(j ** 2 for j in diff) / len(diff), 4)


def calc_mae(original, predicted):
    diff = calc_diff(original, predicted)
    return round(sum(abs(j) for j in diff) / len(diff), 4)


def calc_rsquared(original, predicted):
    lstsumsrs = []
    moriginal = sum(original) / len(original)
    for x in range(len(original)):
        lstsumsrs.append((original[x] - moriginal) ** 2)
    return round(1 - ((calc_mse(original, predicted)) / (sum(lstsumsrs))), 4)


def calc_rmse(original, predicted):
    mse = calc_mse(original, predicted)
    return round(math.sqrt(mse), 4)


def similarity(original, predicted):
    original_norm, predicted_norm = [norm(original), norm(predicted)]
    return {
        "Original": {
            "MAE": calc_mae(original, predicted),
            "MSE": calc_mse(original, predicted),
            "RMSE": calc_rmse(original, predicted),
            "R-Squared": calc_rsquared(original, predicted),
        },
        "Normalized": {
            "MAE": calc_mae(original_norm, predicted_norm),
            "MSE": calc_mse(original_norm, predicted_norm),
            "RMSE": calc_rmse(original_norm, predicted_norm),
            "R-Squared": calc_rsquared(original_norm, predicted_norm),
        },
    }


def sig_regress(xl, yl, normalized=True, rreturn=False, graphing=False):
    # Calculating Z(new x)
    n = len(xl)
    slope_top = (n * (sm(1, n, f"({xl}[@-1])*({yl}[@-1])"))) - sm(
        1, n, f"{xl}[@-1]", fval=1
    ) * sm(1, n, f"{yl}[@-1]", fval=1)
    slope_bottom = (n * (sm(1, n, f"({xl}[@-1]**2)", fval=1))) - (
        sm(1, n, f"{xl}[@-1]")
    ) ** 2
    slope_sig = slope_top / slope_bottom
    yint = (1 / n) * sm(1, n, f"{yl}[@-1]", fval=1) - (
        (slope_sig / n) * sm(1, n, f"{xl}[@-1]", fval=1)
    )
    B0 = yint
    B1 = round(slope_sig, 3)
    if B1 < 0:
        mid = f" - {abs(B1)}"
    else:
        mid = f" + {B1}"
    if rreturn:
        original = yl
        predicted = [
            [math.exp(B0 + (B1 * itr)) * (1 / (1 + math.exp(B0 + (B1 * itr))))][0]
            for itr in yl
        ]
        return [original, predicted]
    elif graphing:
        original = np.linspace(min(xlist), max(xlist), max(ylist))
        expd = np.array(list(map(math.exp, B0 + (B1 * original))))
        predicted = expd * (1 / (1 + expd))
        return [original, predicted]
    else:
        return f"y = exp({B0}{mid}x) * (1 + exp({B0}{mid}x))^-1"


layout = [
    [
        sg.Radio(
            "Quadratic Regression",
            "GraphChoice",
            default=True,
            size=(20, 1),
            enable_events=True,
        ),
    ],
    [
        sg.Radio(
            "Linear Regression",
            "GraphChoice",
            default=True,
            size=(20, 1),
            enable_events=True,
        ),
    ],
    [
        sg.Radio(
            "Logarithmic Regression",
            "GraphChoice",
            default=True,
            size=(20, 1),
            enable_events=True,
        ),
    ],
    [
        sg.Radio(
            "Logistic Regression",
            "GraphChoice",
            default=True,
            size=(20, 1),
            enable_events=True,
        ),
    ],
    [
        sg.Radio(
            "All Graphs",
            "GraphChoice",
            default=True,
            size=(20, 1),
            enable_events=True,
        ),
    ],
    [sg.Checkbox("Randomizer", default=False, key="-RAND VAR-", enable_events=True)],
    [
        sg.Text("Max Value: ", size=(20, 1)),
        sg.Slider(
            (5, 100),
            10,
            1,
            orientation="h",
            size=(20, 15),
            key="-MAXRND SLIDER-",
            enable_events=True,
        ),
    ],
    [
        sg.Text("Amount of Points: ", size=(20, 1)),
        sg.Slider(
            (5, 100),
            10,
            1,
            orientation="h",
            size=(20, 15),
            key="-POINTS SLIDER-",
            enable_events=True,
        ),
    ],
    [sg.Button("Open Coordinate Grid", key="-COORDINATE GRID-")],
    [sg.Button("Generate Graph")],
    [sg.Button("Exit")],
]


def main(METHOD, xl, yl):
    if METHOD == "log":
        sg.Popup(str(log_regress(xl, yl)))
    elif METHOD == "lin":
        sg.Popup(str(lin_regress(xl, yl)))
    elif METHOD == "quad":
        sg.Popup(str(quad_regress(xl, yl)))
    elif METHOD == "sig":
        sg.Popup(str(sig_regress(xl, yl)))
    elif METHOD == "all":
        dctout = {}
        if not any([True if x <= 0 else False][0] for x in xl):
            if log_regress(xl, yl, rreturn=True) != None:
                log_og, log_pr = log_regress(xl, yl, rreturn=True)
                dctout["Logarithmic Regression"] = [log_og, log_pr, log_regress(xl, yl)]

        try:
            lin_og, lin_pr = lin_regress(xl, yl, rreturn=True)
            dctout["Linear Regression"] = [lin_og, lin_pr, lin_regress(xl, yl)]
        except TypeError:
            pass

        try:
            quad_og, quad_pr = quad_regress(xl, yl, rreturn=True)
            dctout["Quadratic Regression"] = [quad_og, quad_pr, quad_regress(xl, yl)]
        except (TypeError, ValueError):
            pass
        try:
            sig_og, sig_pr = sig_regress(xl, yl, rreturn=True)
            dctout["Logistic Regression"] = [sig_og, sig_pr, sig_regress(xl, yl)]
        except TypeError:
            pass

        list(dctout.keys())
        lstvalid = [
            "Quadratic Regression",
            "Linear Regression",
            "Logarithmic Regression",
            "Logistic Regression",
        ]

        try:
            quad_graphx, quadgraphy = quad_regress(xl, yl, graphing=True)
            (quadgraph,) = plt.plot(quad_graphx, quadgraphy)
            quadgraph.set_label("Quadratic Regression")
        except (ValueError, TypeError):
            pass
        sig_graphx, sig_graphy = sig_regress(xl, yl, graphing=True)
        lin_graphx, lin_graphy = lin_regress(xl, yl, graphing=True)
        log_graphx, log_graphy = log_regress(xl, yl, graphing=True)
        (siggraph,) = plt.plot(sig_graphx, sig_graphy)
        (lingraph,) = plt.plot(lin_graphx, lin_graphy)
        (loggraph,) = plt.plot(log_graphx, log_graphy)
        siggraph.set_label("Logistic Regression")
        lingraph.set_label("Linear Regression")
        loggraph.set_label("Logarithmic Regression")
        plt.legend()
        axes = plt.gca()
        axes.set_ylim(
            [min(ylist) - (0.1 * max(ylist)), max(ylist) + (0.1 * max(ylist))]
        )
        plt.show(block=False)

        for x in lstvalid:
            if x not in list(dctout.keys()):
                continue
            tmp1, tmp2 = [[], []]
            for g in range(len(dctout[x][1])):
                try:
                    tmp1.append(dctout[x][0][g])
                    tmp2.append(dctout[x][1][g])
                except IndexError:
                    continue
            dctout[x][0] = tmp1
            dctout[x][1] = tmp2
            dctout2 = {}
            try:
                dctout2 = similarity(dctout[x][0], dctout[x][1])
                dctout2["Equation"] = dctout[x][2]
                dctout[x] = dctout2
            except ValueError:
                continue
        with open("tmp.json", "x+") as jsonoutfile:
            jsonoutfile.write(json.dumps(dctout, indent=4))


global xlist, ylist
rndval, mxrnd = [False, False]


def genrnd(rndval, mxrnd):
    xltmp = []
    yltmp = []
    for x in range(rndval):
        xltmp.append(random.randrange(1, mxrnd))
        yltmp.append(random.randrange(1, mxrnd))
    return [xltmp, yltmp]


window = sg.Window(title="Regression Grapher", layout=layout, margins=(100, 50))
hasclickedbutton = False
valuedict = {
    "0": "quad",
    "1": "lin",
    "2": "log",
    "3": "sig",
    "4": "all",
}
currentbutton = None
set2rand = False
customgrid = False
sc = False
while True:
    event, values = window.read()
    if event == "Exit":
        break
    elif event == "Generate Graph":
        if not hasclickedbutton and currentbutton == None:
            pass
        else:
            if set2rand:
                xlist, ylist = genrnd(
                    int(values["-POINTS SLIDER-"]), int(values["-MAXRND SLIDER-"])
                )
            elif sc != False:
                xlist, ylist = sc
                xlist = list(map(round, xlist))
                ylist = list(map(round, ylist))
            else:
                xlist = [4, 5, 6]
                ylist = [1, 3, 5]
            plt.scatter(np.array(xlist), np.array(ylist))
            foundvalidgraph = False
            main(valuedict[str(currentbutton)], xlist, ylist)
            subprocess.call("del tmp.json", shell=True)
    if event == "-COORDINATE GRID-":
        selectedcoords = gcoords()
        if selectedcoords["XList"] == [] or selectedcoords["YList"] == []:
            sc = False
        else:
            sc = [selectedcoords["XList"], selectedcoords["YList"]]
    if event != None and str(event) in ["0", "1", "2", "3", "4"]:
        hasclickedbutton = True
        currentbutton = str(event)
    if values["-RAND VAR-"]:
        set2rand = True
    if not values["-RAND VAR-"]:
        set2rand = False
window.close()
