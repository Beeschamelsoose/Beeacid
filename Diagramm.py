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
#data = data_full.copy()

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

tick_step_h = 12

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

ax2 = ax1.twinx()

#Nachts is grau
for ns, ne in zip(night_starts, night_ends):
    left  = max(ns, xmin)
    right = min(ne, xmax)
    if right > left:
        ax1.axvspan(left, right, color="gray", alpha=0.25, lw=0)

#Hier Figures definieren

l3,=ax1.plot(delta_t, data["MW AS"]
             ,'.', label='Formic acid', color="green")
ax1.fill_between(delta_t, 
                (data["MW AS"]-data["SD AS"]),
                (data["MW AS"]+data["SD AS"]),
                alpha = 0.5, edgecolor="none", color="lightgreen")
l1,=ax2.plot(delta_t, data["MW TT"],
            '.',label='Outside temperature', color="red")
ax2.fill_between(delta_t,
                data["MW TT"]-data["SD TT"],
                data["MW TT"]+data["SD TT"],
                alpha = 0.5, edgecolor="none", color="orange")
l2,=ax2.plot(delta_t, data["MW RF"],
             '.',label='Outside relative humidity',color="blue")
ax2.fill_between(delta_t, 
                data["MW RF"]-data["SD RF"],
                data["MW RF"]+data["SD RF"],
                alpha = 0.5, edgecolor="none", color="lightblue")
#ax.plot(delta_t,data["t_h"],label='Uhrzeit')
ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_visible(False)

ax2.grid(True)
ax2.grid(True,axis='x',zorder=0)

ax2.set_ylabel("Temperatur [°C]  -  Relative humidity [%]")
ax1.set_ylabel("Formic acid concentration [ppm]")
ax2.set_ylim(-40,100)
ax1.set_ylim(0,800)
#ax1.set_xlim(-5,360)    
#ax2.set_xlim(-5,360)

#X Achse einstellen

axx2 = ax1.secondary_xaxis('top')
ax1.set_xlabel('Time [hours] after application')
axx2.set_xlabel('Calendar days after application ')
labels = [h_to_hhmm(t+offset)for t in ticks]

tick_rel = np.arange(xmin - (xmin % 24), xmax + 24, 24)



axx2.set_xticks(tick_rel)
axx2.set_xticklabels([f"{t/24:.0f}" for t in tick_rel])

ax1.set_xticks(tick_rel)
ax1.set_xticklabels([f"{t:.0f}" for t in tick_rel])

#plt.title("Messwertverläufe nach der Zeit")

lines=[l3,l1,l2]
labels = [line.get_label() for line in lines]

ax2.legend(lines, labels,loc='right')

plt.tight_layout()
plt.show()

