import pandas, csv
import numpy as np
import matplotlib.pyplot as plt
import numpy
dataPath = "../data/"

# reading CSV file
reader = csv.reader(open(dataPath + 'features_2.csv', 'r'), delimiter=',')
data_full = np.array(list(reader))
reader = csv.reader(open(dataPath + 'all_endpoints_with_missing_values_012615.csv', 'r'), delimiter=',')
activity_full = np.array(list(reader))
badValue = 999
# feature names
feature_names = data_full[0, 1:]

# names of the proteins
protein_names = data_full[1:, 0]
protein_names1 = activity_full[1:, 0]
print 'Protein names equality check:', np.array_equal(protein_names1, protein_names)

# names of receptors
receptor_names = activity_full[0, 1:]

# Object-Feature matrix (proteins description)
X = data_full[1:, 1:].astype('double')

# Activity matrix
Y = activity_full[1:, 1:].astype('int16')

# Removing constant features
ind = np.var(X, axis = 0) != 0
X = X[:, ind]
feature_names = feature_names[ind]

# Number of features
N = X.shape[1]

# Normalization
df = pandas.DataFrame(X)
df_norm = (df - df.mean()) / (df.max() - df.min())
X = np.copy(df_norm.as_matrix())
df = pandas.DataFrame(X)


params={'text.usetex' : False,
       'font.size' : 10,
#          'font.family' : 'lmodern',
       'text.latex.unicode': True}
plt.rcParams.update(params)


fig, ax = plt.subplots(figsize=(20,20))
lout=13
ax1 = df.hist(xlabelsize=0,ylabelsize=0, figsize=(15, 15), layout=(lout,lout),sharey=True,ax=ax,grid=False)
#fig.savefig("a.svg", bbox_inches = 'tight')
#ax1.grid(b=False)
for x in range(lout):
    for y in range(lout):
        ax1[x][y].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
        ax1[x][y].tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            right='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
        ax1[x][y].get_xaxis().set_visible(False)
        ax1[x][y].get_yaxis().set_visible(False)
        ax1[x][y].axis('off')
        ax1[x][y].set_axis_off()

plt.tight_layout()
fig.savefig("RealDataDistr.svg", bbox_inches = 'tight')
