Title: Absentee Count
Date: 03/23/2017

```python
import pandas as pd
%matplotlib inline
```


```python
df = pd.read_csv('absentee11xx08xx2016.csv', encoding='latin-1')
```

    /Users/pbaumgartner/anaconda3/lib/python3.5/site-packages/IPython/core/interactiveshell.py:2717: DtypeWarning: Columns (22) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
df['age'].describe()
```




    count    3.188534e+06
    mean     5.230984e+01
    std      1.781151e+01
    min      1.800000e+01
    25%      3.900000e+01
    50%      5.400000e+01
    75%      6.600000e+01
    max      2.630000e+02
    Name: age, dtype: float64




```python
df[df['age'] > 110].sort('age', ascending=0).head()
```

    /Users/pbaumgartner/anaconda3/lib/python3.5/site-packages/ipykernel/__main__.py:1: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      if __name__ == '__main__':





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county_desc</th>
      <th>voter_reg_num</th>
      <th>ncid</th>
      <th>voter_last_name</th>
      <th>voter_first_name</th>
      <th>voter_middle_name</th>
      <th>race</th>
      <th>gender</th>
      <th>age</th>
      <th>voter_street_address</th>
      <th>...</th>
      <th>nc_house_desc</th>
      <th>nc_senate_desc</th>
      <th>ballot_req_delivery_type</th>
      <th>ballot_req_type</th>
      <th>ballot_request_party</th>
      <th>ballot_req_dt</th>
      <th>ballot_send_dt</th>
      <th>ballot_rtn_dt</th>
      <th>ballot_rtn_status</th>
      <th>site_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3130041</th>
      <td>WAYNE</td>
      <td>30079220</td>
      <td>EM73807</td>
      <td>WICKS</td>
      <td>JOHN</td>
      <td>MCCARROLL</td>
      <td>WHITE</td>
      <td>M</td>
      <td>263</td>
      <td>211 W JAMES ST</td>
      <td>...</td>
      <td>NC HOUSE DISTRICT 21</td>
      <td>NC SENATE DISTRICT 7</td>
      <td>IN PERSON</td>
      <td>ONE-STOP</td>
      <td>UNA</td>
      <td>10/31/2016</td>
      <td>10/31/2016</td>
      <td>10/31/2016</td>
      <td>ACCEPTED</td>
      <td>MOUNT OLIVE CIVIC CENTER</td>
    </tr>
    <tr>
      <th>1359010</th>
      <td>HARNETT</td>
      <td>897800</td>
      <td>CA122531</td>
      <td>LANNING</td>
      <td>THOMAS</td>
      <td>NaN</td>
      <td>WHITE</td>
      <td>M</td>
      <td>263</td>
      <td>116 N HILLSIDE DR</td>
      <td>...</td>
      <td>NC HOUSE DISTRICT 53</td>
      <td>NC SENATE DISTRICT 12</td>
      <td>IN PERSON</td>
      <td>ONE-STOP</td>
      <td>LIB</td>
      <td>10/31/2016</td>
      <td>10/31/2016</td>
      <td>10/31/2016</td>
      <td>ACCEPTED</td>
      <td>HARNETT COUNTY BOARD OF ELECTIONS OFFICE</td>
    </tr>
    <tr>
      <th>2719430</th>
      <td>VANCE</td>
      <td>9573</td>
      <td>EG11167</td>
      <td>MOORE</td>
      <td>HELEN</td>
      <td>LOWRY</td>
      <td>WHITE</td>
      <td>F</td>
      <td>190</td>
      <td>1214  ANNE ST</td>
      <td>...</td>
      <td>NC HOUSE DISTRICT 32</td>
      <td>NC SENATE DISTRICT 4</td>
      <td>IN PERSON</td>
      <td>ONE-STOP</td>
      <td>DEM</td>
      <td>11/04/2016</td>
      <td>11/04/2016</td>
      <td>11/04/2016</td>
      <td>ACCEPTED</td>
      <td>HENDERSON OPERATIONS CENTER</td>
    </tr>
    <tr>
      <th>1675291</th>
      <td>MCDOWELL</td>
      <td>76597</td>
      <td>CT44511</td>
      <td>HARRIS</td>
      <td>JACKIE</td>
      <td>RAY</td>
      <td>WHITE</td>
      <td>M</td>
      <td>154</td>
      <td>427 E MAIN ST</td>
      <td>...</td>
      <td>NC HOUSE DISTRICT 85</td>
      <td>NC SENATE DISTRICT 47</td>
      <td>IN PERSON</td>
      <td>ONE-STOP</td>
      <td>UNA</td>
      <td>10/28/2016</td>
      <td>10/28/2016</td>
      <td>10/28/2016</td>
      <td>ACCEPTED</td>
      <td>OLD FORT LIBRARY</td>
    </tr>
    <tr>
      <th>1000402</th>
      <td>FORSYTH</td>
      <td>30273599</td>
      <td>BN493255</td>
      <td>ROBBS</td>
      <td>SHARRON</td>
      <td>LOUISE</td>
      <td>BLACK or AFRICAN AMERICAN</td>
      <td>F</td>
      <td>149</td>
      <td>2364 S HIGHWAY 66</td>
      <td>...</td>
      <td>NC HOUSE DISTRICT 75</td>
      <td>NC SENATE DISTRICT 31</td>
      <td>IN PERSON</td>
      <td>ONE-STOP</td>
      <td>DEM</td>
      <td>11/04/2016</td>
      <td>11/04/2016</td>
      <td>11/04/2016</td>
      <td>ACCEPTED</td>
      <td>KERNERSVILLE SENIOR CENTER</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python

```
