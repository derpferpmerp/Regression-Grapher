import PySimpleGUI as sg


sg.theme("Black")


def TableSimulation():
    sg.popup_quick_message(
        "Creating Grid", auto_close=True, non_blocking=True, font="Default 18"
    )
    sg.set_options(element_padding=(0, 0))

    MAX_ROWS = 100
    MAX_COL = 2

    columm_layout = [
        [sg.Text(str(i), size=(4, 1), justification="right")]
        + [
            sg.InputText(
                size=(10, 1),
                pad=(1, 1),
                border_width=0,
                justification="center",
                key=str((i, j)),
            )
            for j in range(MAX_COL)
        ]
        for i in range(MAX_ROWS)
    ]
    layout = [
        [sg.Text("Coordinate Grid", font="Any 18")],
        [sg.Text("Enter the X and Y Coordinates")],
        [sg.Col(columm_layout, size=(800, 600), scrollable=True)],
    ]

    window = sg.Window("Table", layout, return_keyboard_events=True)
    dctwindow = []
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            return dctwindow
        for x in range(MAX_COL):
            for g in range(MAX_ROWS):
                location = (g, x)
                window[str((g, x))]
                if values[str((g, x))] not in ["", None]:
                    dctwindow.append(
                        {
                            "X": str(location)[1:-1].split(",")[0],
                            "Y": str(location)[1:-1].split(",")[1],
                            "Value": f"{values[str((g, x))]}",
                        }
                    )
    window.close()


def main():
    tablevalues = TableSimulation()
    pruned = []
    replaceable = {}
    for x in tablevalues:
        if x not in pruned:
            replaceable[str((x["X"], x["Y"]))] = x["Value"]

    coordsdict = {}
    for x in list(replaceable.keys()):
        list(replaceable.keys()).index(x)
        try:
            xcoord = int(str(x)[1:-1].replace(" ", "").replace("'", "").split(",")[0])
            ycoord = int(str(x)[1:-1].replace(" ", "").replace("'", "").split(",")[1])
            # dctout.append({
            # 	"X": xcoord,
            # 	"Y": int(str(x)[1:-1].replace(' ','').replace('\'','').split(",")[1]),
            # 	"Value": float(replaceable[x])
            # })
            if str(xcoord) not in list(coordsdict.keys()):
                coordsdict[str(xcoord)] = {"X": replaceable[x]}
            if ycoord == 1 and str(xcoord) in list(coordsdict.keys()):
                coordsdict[str(xcoord)]["Y"] = replaceable[x]
        except (ValueError, TypeError):
            continue
    outdict2 = {"XList": [], "YList": []}
    for x in list(coordsdict.values()):
        if len(x) == 2:
            outdict2["XList"].append(float(x["X"]))
            outdict2["YList"].append(float(x["Y"]))
    return outdict2
