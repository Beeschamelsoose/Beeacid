import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ezodf


path = "Werte_ohne.ods"
doc = ezodf.opendoc(path)

sheet = doc.sheets[0]

rows = []
for row in sheet.rows():
    cells = [cell.plaintext() for cell in row]
    rows.append(cells)
#alles einlen
data_full = pd.DataFrame(rows[1:],columns=rows[0])
#abschneiden bei diskontinuirät
data = data_full.iloc[:366].copy()

for col in data.columns [1:]:
    data[col] = pd.to_numeric(data[col],errors="coerce")

#verkürzen für diagramme
delta_t = data["Zeit relativ"]

h, m, s=map(float, data["MW Zeit zirkadian"].iloc[0].split(":"))
t_0 = h + m/60 + s/3600
##Offset für Uhrzeit 0
offset = t_0-delta_t.iloc[0]                       
print("Offset [h]:", offset)

#Wrapper für Tageszeit in HH:tttttt und hh:mm:ss
delta_s = (delta_t+offset)*3600
s_mod = delta_s %(24*3600)
data["t_h"]=s_mod/3600.0

hh= (s_mod//3600).astype(int)
mm = ((s_mod % 3600) // 60).astype(int)
ss= (s_mod%60).astype(int)


data["time"] = (
    hh.map("{:02d}".format) + ":" + mm.map("{:02d}".format) + ":" + ss.map("{:02d}".format)
)

print(data.head())

def h_to_hhmm(h):
    total_sec = int(np.floor((h%24))*3600+1e-6)
    hh = total_sec //3600
    mm = (total_sec % 3600) // 60
    return f"{hh:02d}:{mm:02d}"

tick_step_h = 6

xmin = delta_t.min()
xmax = delta_t.max()

k_start = int(np.ceil((xmin + offset)/tick_step_h))
k_end = int(np.floor((xmax+offset)/tick_step_h))

ticks = tick_step_h*np.arange(k_start,k_end+1)-offset

n_start = int(np.floor((xmin + offset - 18) / 24))
n_end   = int(np.ceil((xmax + offset - 18) / 24))

night_starts = 18 + 24*np.arange(n_start, n_end + 1) - offset
night_ends   = night_starts + 12



#plot
fig, ax1 =plt.subplots()
axisstretch = 0.1
ax2 = ax1.twinx()

#Nachts is grau
for ns, ne in zip(night_starts, night_ends):
    left  = max(ns, xmin)
    right = min(ne, xmax)
    if right > left:
        ax1.axvspan(left, right, color="gray", alpha=0.25, lw=0)

#Hier Figures definieren
l1,=ax1.plot(delta_t, data["MW TT"],'-x',label='Temperaturmittel')
ax1.fill_between(delta_t, data["MW TT"]-data["SD TT"],data["MW TT"]+data["SD TT"],alpha = 0.2, edgecolor="none")
l2,=ax1.plot(delta_t, data["MW RF"],'-x',label='Feuchtigkeitsmittel')
ax1.fill_between(delta_t, data["MW RF"]-data["SD RF"],data["MW RF"]+data["SD RF"],alpha = 0.2, edgecolor="none")
l3,=ax2.plot(delta_t, data["MW AS"],'-x', label='Ameisensäuremittel', color="green")
ax2.fill_between(delta_t, (data["MW AS"]-data["SD AS"]),(data["MW AS"]+data["SD AS"]),alpha = 0.2, edgecolor="none")
#ax.plot(delta_t,data["t_h"],label='Uhrzeit')
ax1.grid(True)
ax1.set_ylabel("T [°C] / rel. Luftfeuchte [%h]")
ax2.set_ylabel("Konzentration Ameisensäure")

labels = [h_to_hhmm(t+offset)for t in ticks]
ax1.set_xticks(ticks)
ax1.set_xticklabels(labels, rotation=90)

plt.tight_layout()
lines=[l1,l2,l3]
labels = [line.get_label() for line in lines]
plt.legend(lines, labels)
plt.show()

