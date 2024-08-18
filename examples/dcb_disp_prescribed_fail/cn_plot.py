from pandas import read_csv
import matplotlib.pyplot as plt
df = read_csv('cn.csv')

ax = plt.subplot()
ax.plot(df.index,df.cn)
plt.show()