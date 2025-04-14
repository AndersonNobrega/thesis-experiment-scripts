import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import datetime as dt
import time
import seaborn as sns

custom = {"axes.edgecolor": "black"}
sns.set_style("whitegrid", rc=custom)

file = open('/home/anderson/code_projects/mestrado/data/casa_jefferson/casa_jefferson_train.dat', 'rb')
arr = pickle.load(file)

print(arr.shape)

max_value = arr[:, 1, 0].max()

plt.subplots(figsize=(12, 8))

# dates=[dt.datetime.fromtimestamp(ts) for ts in arr[:, 0, 0]]
# datenums=md.date2num(dates)

# ax = plt.gca()
# xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
# ax.xaxis.set_major_formatter(xfmt)

plt.subplot(211)
plt.ylabel("Consumo - Potência Ativa (W)")
plt.plot(arr[:, 1, 0])

plt.subplot(212)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=25)
plt.grid(True, zorder=0)
plt.title('Consumo Total')
plt.ylabel("Consumo - Potência Reativa (W)")
plt.ylim(0, max_value + 200)
plt.plot(arr[:, 1, 1])

plt.show()