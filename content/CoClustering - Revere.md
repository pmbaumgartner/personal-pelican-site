Title: Biclustering Workflow
Date: 03/29/17
Tags: python, sklearn, pandas, clustering
Category: Workflow

Biclustering is a helpful clustering procedure that derives cluster membership for both rows and columns. Clustering rows and columns individually is fairly common. I view biclustering as an extension that allows me to ask: "If these columns are clustered together, which rows (if any) point towards that clustering? Accordingly, if any rows are clustered together, which column values might tell me why they're clustered?"

I have found it helpful with:

- Incidence Matrix / Cooccurence data (typically found before converting something into a graph/network structure)
- Text data (the [original paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.3011&rep=rep1&type=pdf) on coclustering does this)

The example below performs coclustering on the awesome "Paul Revere dataset" originally analyzed in an [amazing blog post](https://kieranhealy.org/blog/archives/2013/06/09/using-metadata-to-find-paul-revere/) by [Kieran Healy](https://kieranhealy.org/).

## References:
> Biclustering algorithms simultaneously cluster rows and columns of a data matrix. These clusters of rows and columns are known as biclusters. Each determines a submatrix of the original data matrix with some desired properties.

[sklearn - Biclustering](http://scikit-learn.org/stable/modules/biclustering.html#biclustering)


```python
import re

from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster.bicluster import SpectralCoclustering

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
```

## Data Import


```python
data = pd.read_csv('https://raw.githubusercontent.com/kjhealy/revere/master/data/PaulRevereAppD.csv', index_col=0)
```


```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StAndrewsLodge</th>
      <th>LoyalNine</th>
      <th>NorthCaucus</th>
      <th>LongRoomClub</th>
      <th>TeaParty</th>
      <th>BostonCommittee</th>
      <th>LondonEnemies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adams.John</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Adams.Samuel</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Allen.Dr</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Appleton.Nathaniel</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ash.Gilbert</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.index = [i.replace('.', ', ') for i in data.index]
```


```python
nrows, ncolumns = data.shape
print("rows:", nrows, "columns:", ncolumns)
```

    rows: 254 columns: 7


## Fit Model


```python
cocluster = SpectralCoclustering(n_clusters=6, random_state=666, svd_method='arpack')
cocluster.fit(data)
```




    SpectralCoclustering(init='k-means++', mini_batch=False, n_clusters=6,
               n_init=10, n_jobs=1, n_svd_vecs=None, random_state=666,
               svd_method='arpack')



## Explore Results


```python
cluster_data = []
for cluster in range(cocluster.n_clusters):
    row_ix, col_ix = cocluster.get_indices(cluster)
    data_subset = data.ix[row_ix, col_ix]
    c_rows, c_cols = data_subset.shape
    orgs = data_subset.columns
    names = data_subset.index
    print("Cluster: ", cluster, "\tMembers:", c_rows, "\tOrgs:", c_cols)
    print("Orgs:   ", '; '.join(["{}".format(org) for org in orgs]))
    print("Members:", '; '.join(["{}".format(name) for name in names]))
    print("---")
    cluster = {'Cluster' : cluster, 'Members' : '; '.join(["{}".format(name) for name in names]), 'Orgs' : '; '.join(["{}".format(org) for org in orgs])}
    cluster_data.append(cluster)
```

    Cluster:  0 	Members: 68 	Orgs: 3
    Orgs:    LongRoomClub; BostonCommittee; LondonEnemies
    Members: Adams, John; Adams, Samuel; Appleton, Nathaniel; Austin, Benjamin; Austin, Samuel; Baldwin, Cyrus; Boyer, Peter; Boynton, Richard; Brackett, Jos; Bradford, John; Brimmer, Herman; Brimmer, Martin; Broomfield, Henry; Brown, Enoch; Brown, John; Cheever, Ezekiel; Church, Benjamin; Cooper, William; Davis, Caleb; Davis, Edward; Davis, William; Dawes, Thomas; Dennie, William; Dexter, Samuel; Fleet, Thomas; Foster, Bos; Gill, Moses; Greenleaf, Joseph; Greenleaf, William; Greenough, Newn; Hancock, Eben; Hancock, John; Hill, Alexander; Hopkins, Caleb; Isaac, Pierce; Ivers, James; Jarvis, Charles; Johnston, Eben; Lambert, John; Mackay, William; Marshall, Thomas; Marson, John; Mason, Jonathan; Noyces, Nat; Otis, James; Parkman, Elias; Partridge, Sam; Phillips, Samuel; Phillips, William; Pierpont, Robert; Pitts, John; Pitts, Samuel; Powell, William; Prince, Job; Pulling, John; Quincy, Josiah; Roylson, Thomas; Ruddock, Abiel; Sweetser, John; Tyler, Royall; Vernon, Fortesque; Waldo, Benjamin; Wendell, Oliver; Whitwell, Samuel; Whitwell, William; Williams, Jonathan; Winslow, John; Winthrop, John
    ---
    Cluster:  1 	Members: 81 	Orgs: 1
    Orgs:    TeaParty
    Members: Barnard, Samuel; Bolter, Thomas; Bradlee, David; Bradlee, Josiah; Bradlee, Nathaniel; Bradlee, Thomas; Bewer, James; Bruce, Stephen; Burton, Benjamin; Campbell, Nicholas; Clarke, Benjamin; Cochran, John; Colesworthy, Gilbert; Collier, Gershom; Crane, John; Davis, Robert; Dolbear, Edward; Eaton, Joseph; Eckley, Unknown; Etheridge, William; Fenno, Samuel; Foster, Samuel; Frothingham, Nathaniel; Gammell, John; Gore, Samuel; Greene, Nathaniel; Hammond, Samuel; Hendley, William; Hewes, George; Hicks, John; Hobbs, Samuel; Hooton, John; Howard, Samuel; Howe, Edward; Hunnewell, Jonathan; Hunnewell, Richard; Hunstable, Thomas; Hunt, Abraham; Ingersoll, Daniel; Kinnison, David; Lee, Joseph; Lincoln, Amos; Loring, Matthew; Machin, Thomas; MacKintosh, Capt; MacNeil, Archibald; May, John; Melville, Thomas; Moore, Thomas; Morse, Anthony; Mountford, Joseph; Newell, Eliphelet; Palmer, Joseph; Parker, Jonathan; Payson, Joseph; Peters, John; Pierce, William; Pitts, Lendall; Porter, Thomas; Prentiss, Henry; Prince, John; Purkitt, Henry; Randall, John; Roby, Joseph; Russell, John; Russell, William; Sessions, Robert; Shed, Joseph; Simpson, Benjamin; Slater, Peter; Spear, Thomas; Sprague, Samuel; Spurr, John; Starr, James; Stearns, Phineas; Stevens, Ebenezer; Wheeler, Josiah; Williams, Jeremiah; Williams, Thomas; Willis, Nathaniel; Wyeth, Joshua
    ---
    Cluster:  2 	Members: 46 	Orgs: 1
    Orgs:    StAndrewsLodge
    Members: Ash, Gilbert; Bell, William; Blake, Increase; Bray, George; Brown, Hugh; Burbeck, Edward; Burbeck, William; Cailleteau, Edward; Callendar, Elisha; Chipman, Seth; Collins, Ezra; Deshon, Moses; Doyle, Peter; Ferrell, Ambrose; Flagg, Josiah; Gould, William; Graham, James; Gray, Wait; Ham, William; Hitchborn, Nathaniel; Hoffins, John; Inglish, Alexander; Jarvis, Edward; Jefferds, Unknown; Jenkins, John; Kerr, Walter; Lewis, Phillip; Marett, Phillip; Marlton, John; McAlpine, William; Milliken, Thomas; Moody, Samuel; Nicholls, Unknown; Obear, Israel; Palfrey, William; Phillips, John; Potter, Edward; Pulling, Richard; Seward, James; Sloper, Ambrose; Stanbridge, Henry; Tabor, Philip; Webb, Joseph; Webster, Thomas; Whitten, John; Wingfield, William
    ---
    Cluster:  3 	Members: 12 	Orgs: 0
    Orgs:    
    Members: Bass, Henry; Chase, Thomas; Collson, Adam; Condy, JamesFoster; Cooper, Samuel; Eayres, Joseph; Grant, Moses; Molineux, William; Proctor, Edward; Story, Elisha; Swan, James; Young, Thomas
    ---
    Cluster:  4 	Members: 37 	Orgs: 1
    Orgs:    NorthCaucus
    Members: Allen, Dr; Avery, John; Ballard, John; Barber, Nathaniel; Boit, John; Breck, William; Burt, Benjamin; Cazneau, Capt; Chadwell, Mr; Champney, Caleb; Chrysty, Thomas; Edes, Benjamin; Emmes, Samuel; Hickling, William; Hitchborn, Thomas; Holmes, Nathaniel; Hoskins, William; Johonnott, Gabriel; Kent, Benjamin; Kimball, Thomas; Lowell, John; Matchett, John; Merrit, John; Morton, Perez; Palms, Richard; Pearce, IsaacJun; Pearce, Isaac; Peck, Thomas; Sharp, Gibbens; Sigourney, John; Stoddard, Asa; Stoddard, Jonathan; Symmes, Eben; Symmes, John; Tileston, Thomas; Warren, Joseph; White, Samuel
    ---
    Cluster:  5 	Members: 10 	Orgs: 1
    Orgs:    LoyalNine
    Members: Barrett, Samuel; Cleverly, Stephen; Crafts, Thomas; Field, Joseph; Peck, Samuel; Revere, Paul; Smith, John; Trott, George; Urann, Thomas; Welles, Henry
    ---



```python
cluster_info = pd.DataFrame(cluster_data)
```


```python
cluster_info
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cluster</th>
      <th>Members</th>
      <th>Orgs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Adams, John; Adams, Samuel; Appleton, Nathanie...</td>
      <td>LongRoomClub; BostonCommittee; LondonEnemies</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Barnard, Samuel; Bolter, Thomas; Bradlee, Davi...</td>
      <td>TeaParty</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Ash, Gilbert; Bell, William; Blake, Increase; ...</td>
      <td>StAndrewsLodge</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Bass, Henry; Chase, Thomas; Collson, Adam; Con...</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Allen, Dr; Avery, John; Ballard, John; Barber,...</td>
      <td>NorthCaucus</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Barrett, Samuel; Cleverly, Stephen; Crafts, Th...</td>
      <td>LoyalNine</td>
    </tr>
  </tbody>
</table>
</div>



## Format and Sort DataFrame with MultiIndex
Since rows and columns now have hierarchical information with cluster membership, we can reindex the dataframe to take this into account and index easily on cluster. This should allow us to see the block cluster structure along the diagonal.

This is adapted from an example in the [scikit-learn docs](http://scikit-learn.org/stable/auto_examples/bicluster/bicluster_newsgroups.html
) and takes advantage of using a `MultiIndex` in pandas. I also think it's a lot more readable (both code and output) than figuring out the indices with unlabeled numpy arrays.


```python
data_mi = data.copy()

cocluster_rows = list(zip(cocluster.row_labels_, data.index))
row_index = pd.MultiIndex.from_tuples(cocluster_rows, names=['cluster', 'person'])
data_mi.index = row_index
data_mi = data_mi.sortlevel(axis=0)

cocluster_columns = list(zip(cocluster.column_labels_, data.columns))
col_index = pd.MultiIndex.from_tuples(cocluster_columns, names=['cluster', 'org'])
data_mi.columns = col_index
data_mi = data_mi.sortlevel(axis=1)
```


```python
data_mi.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>cluster</th>
      <th colspan="3" halign="left">0</th>
      <th>1</th>
      <th>2</th>
      <th>4</th>
      <th>5</th>
    </tr>
    <tr>
      <th></th>
      <th>org</th>
      <th>BostonCommittee</th>
      <th>LondonEnemies</th>
      <th>LongRoomClub</th>
      <th>TeaParty</th>
      <th>StAndrewsLodge</th>
      <th>NorthCaucus</th>
      <th>LoyalNine</th>
    </tr>
    <tr>
      <th>cluster</th>
      <th>person</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">0</th>
      <th>Adams, John</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Adams, Samuel</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Appleton, Nathaniel</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Austin, Benjamin</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Austin, Samuel</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Baldwin, Cyrus</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Boyer, Peter</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Boynton, Richard</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Brackett, Jos</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Bradford, John</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Verify Coclustering structure
"The resulting bicluster structure is block-diagonal, since each row and each column belongs to exactly one bicluster." - [sklearn SpectralCoclustering Docs](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.bicluster.SpectralCoclustering.html)


```python
plt.matshow(data_mi.values.T, cmap=plt.cm.Blues, aspect='auto', origin='lower')
plt.xlabel('row index')
plt.ylabel('column index')
plt.show()
# I plotted the transpose so it doesn't give us a long image
```


![png]({filename}/images/CoClustering-Revere_18_0.png)


## Brief Analysis
Let's take a look and see where the actual Loyal Nine members ended up. The organization ended up in our `Cluster 5`, but how many members of that bicluster match?

Loyal Nine [members](https://books.google.com/books?id=iNgNCgAAQBAJ&pg=PA26#v=onepage&q&f=false
):

- John Avery, a distiller and club secretary;
- John Smith and Stephen Cleverly, both braziers;
- Thomas Crafts, a printer;
- Benjamin Edes, who along with John Gill produced the important Boston Gazette;
- Thomas Chase, a distiller;
- Joseph Field, a ship's captain;
- George Trott, a jeweler;
- Henry Bass, a merchant related to Samuel Adams


```python
loyal_nine = ['Avery, John', 'Smith, John', 'Cleverly, Stephen', 'Crafts, Thomas', 'Edes, Benjamin', 'Chase, Thomas',
              'Field, Joseph', 'Trott, George', 'Bass, Henry']
```


```python
data_mi[data_mi.index.get_level_values('person').isin(loyal_nine)]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>cluster</th>
      <th colspan="3" halign="left">0</th>
      <th>1</th>
      <th>2</th>
      <th>4</th>
      <th>5</th>
    </tr>
    <tr>
      <th></th>
      <th>org</th>
      <th>BostonCommittee</th>
      <th>LondonEnemies</th>
      <th>LongRoomClub</th>
      <th>TeaParty</th>
      <th>StAndrewsLodge</th>
      <th>NorthCaucus</th>
      <th>LoyalNine</th>
    </tr>
    <tr>
      <th>cluster</th>
      <th>person</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>Bass, Henry</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Chase, Thomas</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">4</th>
      <th>Avery, John</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Edes, Benjamin</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">5</th>
      <th>Cleverly, Stephen</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Crafts, Thomas</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Field, Joseph</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Smith, John</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Trott, George</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



In `Cluster 5` we recovered all individuals who were *exclusively* members of the Loyal Nine, plus Thomas Crafts. `Cluster 4` pulled two Loyal Nine members also part of the North Caucus and London Enemies, and `Cluster 3` holds members that were additionally part of the North Cacus, Tea Party, and London Enemies each. Also note Cluster 3 contains no organizations -- it's members are part of several organizations.

## Export to Excel
It's helpful to export this to Excel because the MultiIndex formats nicely into merged cells and retains the hierarchical indexing structure.

The data is on the `data` sheet, and cluster details are on the `clusters` sheet.


```python
data_sheets = [(data_mi, 'data'),
               (cluster_info, 'clusters')]

filename = 'bicluster.xlsx'
writer = pd.ExcelWriter(filename, engine='xlsxwriter')
for data in data_sheets:
    data[0].to_excel(writer, sheet_name=data[1])
writer.save()
```
