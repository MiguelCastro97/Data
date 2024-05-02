# FIFA 2021 Dataset

In this project I will clean the FIFA 2021 dataset I found in Kaggle: https://www.kaggle.com/datasets/yagunnersya/fifa-21-messy-raw-dataset-for-cleaning-exploring/data

These are the excersices I'll do:

* Convert the height and weight columns to numerical forms
* Remove the unnecessary newline characters from all columns that have them.
* Based on the 'Joined' column, check which players have been playing at a club for more than 10 years
* 'Value', 'Wage' and "Release Clause' are string columns. Convert them to numbers.
* Some columns have 'star' characters. Strip those columns of these stars and make the columns numerical
* Which players are highly valuable but still underpaid (on low wages)? 


## Import Libraries


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#While I was working on the project, I changed this setting to None, which will showed all rows of any given
#table, but for the final result I restricted the number to 20 to not end up with 4000 pages or more
pd.set_option('display.max_rows', 20)

#This will make pandas show all columns of the dataframe
pd.set_option('display.max_columns', 6)

#I changed the color of the plots to have a dark theme
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'

```

## Import dataset and create Data Frame

Upon inspecting the dataset I decided that I won't need the photourl column


```python
#Data frame which will be modify
fifa_df = pd.read_csv('fifa21_raw_data.csv', low_memory=False)

#Dataframe for data inspection, it won't be changed
fifa_df_original = pd.read_csv('fifa21_raw_data.csv', low_memory=False)
fifa_df.drop(columns='photoUrl', inplace=True)
fifa_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LongName</th>
      <th>playerUrl</th>
      <th>Nationality</th>
      <th>...</th>
      <th>DEF</th>
      <th>PHY</th>
      <th>Hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lionel Messi</td>
      <td>http://sofifa.com/player/158023/lionel-messi/2...</td>
      <td>Argentina</td>
      <td>...</td>
      <td>38</td>
      <td>65</td>
      <td>\n372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C. Ronaldo dos Santos Aveiro</td>
      <td>http://sofifa.com/player/20801/c-ronaldo-dos-s...</td>
      <td>Portugal</td>
      <td>...</td>
      <td>35</td>
      <td>77</td>
      <td>\n344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jan Oblak</td>
      <td>http://sofifa.com/player/200389/jan-oblak/210005/</td>
      <td>Slovenia</td>
      <td>...</td>
      <td>52</td>
      <td>90</td>
      <td>\n86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kevin De Bruyne</td>
      <td>http://sofifa.com/player/192985/kevin-de-bruyn...</td>
      <td>Belgium</td>
      <td>...</td>
      <td>64</td>
      <td>78</td>
      <td>\n163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Neymar da Silva Santos Jr.</td>
      <td>http://sofifa.com/player/190871/neymar-da-silv...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>36</td>
      <td>59</td>
      <td>\n273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Robert Lewandowski</td>
      <td>http://sofifa.com/player/188545/robert-lewando...</td>
      <td>Poland</td>
      <td>...</td>
      <td>43</td>
      <td>82</td>
      <td>\n182</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kylian Mbappé</td>
      <td>http://sofifa.com/player/231747/kylian-mbappe/...</td>
      <td>France</td>
      <td>...</td>
      <td>39</td>
      <td>76</td>
      <td>\n646</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alisson Ramses Becker</td>
      <td>http://sofifa.com/player/212831/alisson-ramses...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>51</td>
      <td>91</td>
      <td>\n79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mohamed Salah</td>
      <td>http://sofifa.com/player/209331/mohamed-salah/...</td>
      <td>Egypt</td>
      <td>...</td>
      <td>45</td>
      <td>75</td>
      <td>\n164</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sadio Mané</td>
      <td>http://sofifa.com/player/208722/sadio-mane/210...</td>
      <td>Senegal</td>
      <td>...</td>
      <td>44</td>
      <td>76</td>
      <td>\n170</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 76 columns</p>
</div>



## Height and Weight Columns

Let's inspect the Height and Weight columns


```python
fifa_df[['Height', 'Weight']].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5'7"</td>
      <td>159lbs</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6'2"</td>
      <td>183lbs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6'2"</td>
      <td>192lbs</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5'11"</td>
      <td>154lbs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5'9"</td>
      <td>150lbs</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6'0"</td>
      <td>176lbs</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5'10"</td>
      <td>161lbs</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6'3"</td>
      <td>201lbs</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5'9"</td>
      <td>157lbs</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5'9"</td>
      <td>152lbs</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6'4"</td>
      <td>203lbs</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6'2"</td>
      <td>187lbs</td>
    </tr>
    <tr>
      <th>12</th>
      <td>6'1"</td>
      <td>185lbs</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6'6"</td>
      <td>212lbs</td>
    </tr>
    <tr>
      <th>14</th>
      <td>6'4"</td>
      <td>203lbs</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6'1"</td>
      <td>179lbs</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6'0"</td>
      <td>181lbs</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5'8"</td>
      <td>154lbs</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5'6"</td>
      <td>154lbs</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5'9"</td>
      <td>161lbs</td>
    </tr>
  </tbody>
</table>
</div>



As we can see the Height column is in feet-inches with a format of 0-9"0-9', while the weight column is in pounds with the format 0-9lbs.

I would like to see if there is any entry that doesn't match this format, or if we have any missing values to take into account


```python
#With the ReEx rule I looked for characters different than 0 to 9, ' and "
height_diff = fifa_df['Height'].str.contains(r'[^0-9\'\"]')
fifa_df['Height'].loc[height_diff].head(20)
```




    Series([], Name: Height, dtype: object)



There are no values that don't fit the format. Let's see if that is true for the weight column


```python
#We follow the same logic as the height column
weight_diff = fifa_df['Weight'].str.contains('[^0-9lbs]')
fifa_df['Weight'].loc[weight_diff].head(20)
```




    Series([], Name: Weight, dtype: object)



As we can see there are no entries that don't fit the format

### Changing Units 

I am not fan of feet inches and pounds for measuring, so I will turn both columns into international units.

Let's start with the Height column:
 - I am going to retrieve all the feet values and put them into a series
 - I will do the same for the inches values
 - Convert into meters
 - Sum them


```python
#I extracted only the first number in the column
feet = fifa_df['Height'].str.extract(r'([0-9])')
#I extracted the number immediatly after the (') character
inches = fifa_df['Height'].str.extract(r'\'([0-9])')
#I concated the dataframes to show them together
feet_inches_df= pd.concat([fifa_df['Height'],feet,inches], axis=1)
#Rename the columns for show purposes
feet_inches_df.columns = ['Original','Feet','Inches']
feet_inches_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Original</th>
      <th>Feet</th>
      <th>Inches</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5'7"</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6'2"</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6'2"</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5'11"</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5'9"</td>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6'0"</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5'10"</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6'3"</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5'9"</td>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5'9"</td>
      <td>5</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
#I defined two functions which I am going to map to feet and inches dataframes
def feet_to_meters(item):
    return round(int(item) * .3048, 2)
def inches_to_meters(item):
    return round(int(item) * 0.0254, 2)
#I applied those functions to each element
meters_feet = feet.applymap(feet_to_meters)
meters_inches = inches.applymap(inches_to_meters)
#I added them up together
height_meters = meters_feet.add(meters_inches)
height_meters.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.75</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.83</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.55</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.91</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.75</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.75</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have our dataframe with the height values in meters it's time to substitute it in the original Height column and rename it to let the reader know it is in meters


```python
fifa_df['Height'] = height_meters
fifa_df.rename(columns={'Height': 'Height(mts)'}, inplace=True)
```

Let's now check if our Height(mts) column is a nummeric column


```python
fifa_df['Height(mts)'].dtypes
```




    dtype('float64')



Let's move on to the Weight column:

 - I am going to get rid of all the lbs strings in the entries
 - Convert the values into kg


```python
weight_num = fifa_df['Weight'].str.strip('lbs')
weight_num.head(10)
```




    0    159
    1    183
    2    192
    3    154
    4    150
    5    176
    6    161
    7    201
    8    157
    9    152
    Name: Weight, dtype: object



Now that I have got rid of the 'lbs' string in the column and change its type to int, I will convert all values from pounds to kilograms.

Note: For some reason, when applying the method str.extract() you will get a Dataframe, while from a str.strip() you will get a Series, this is important because there is no way to use applymap() to a series object, only to a Dataframe, I had to use the map() function instead.


```python
def pounds_kilograms(item):
    return int(int(item) * 0.4535)
weight_kg = weight_num.map(pounds_kilograms)
weight_kg.head(10)
```




    0    72
    1    82
    2    87
    3    69
    4    68
    5    79
    6    73
    7    91
    8    71
    9    68
    Name: Weight, dtype: int64



Now it is time to assign our values to the 'Weight' column in the original Dataframe


```python
fifa_df['Weight'] = weight_kg
fifa_df.rename(columns = {'Weight': 'Weight(kg)'}, inplace=True)
fifa_df.columns
```




    Index(['LongName', 'playerUrl', 'Nationality', 'Positions', 'Name', 'Age',
           '↓OVA', 'POT', 'Team & Contract', 'ID', 'Height(mts)', 'Weight(kg)',
           'foot', 'BOV', 'BP', 'Growth', 'Joined', 'Loan Date End', 'Value',
           'Wage', 'Release Clause', 'Attacking', 'Crossing', 'Finishing',
           'Heading Accuracy', 'Short Passing', 'Volleys', 'Skill', 'Dribbling',
           'Curve', 'FK Accuracy', 'Long Passing', 'Ball Control', 'Movement',
           'Acceleration', 'Sprint Speed', 'Agility', 'Reactions', 'Balance',
           'Power', 'Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots',
           'Mentality', 'Aggression', 'Interceptions', 'Positioning', 'Vision',
           'Penalties', 'Composure', 'Defending', 'Marking', 'Standing Tackle',
           'Sliding Tackle', 'Goalkeeping', 'GK Diving', 'GK Handling',
           'GK Kicking', 'GK Positioning', 'GK Reflexes', 'Total Stats',
           'Base Stats', 'W/F', 'SM', 'A/W', 'D/W', 'IR', 'PAC', 'SHO', 'PAS',
           'DRI', 'DEF', 'PHY', 'Hits'],
          dtype='object')



We have successfuly change the type and units of our columns Height and Weight. This is how our table looks like afterwards.


```python
fifa_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LongName</th>
      <th>playerUrl</th>
      <th>Nationality</th>
      <th>...</th>
      <th>DEF</th>
      <th>PHY</th>
      <th>Hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lionel Messi</td>
      <td>http://sofifa.com/player/158023/lionel-messi/2...</td>
      <td>Argentina</td>
      <td>...</td>
      <td>38</td>
      <td>65</td>
      <td>\n372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C. Ronaldo dos Santos Aveiro</td>
      <td>http://sofifa.com/player/20801/c-ronaldo-dos-s...</td>
      <td>Portugal</td>
      <td>...</td>
      <td>35</td>
      <td>77</td>
      <td>\n344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jan Oblak</td>
      <td>http://sofifa.com/player/200389/jan-oblak/210005/</td>
      <td>Slovenia</td>
      <td>...</td>
      <td>52</td>
      <td>90</td>
      <td>\n86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kevin De Bruyne</td>
      <td>http://sofifa.com/player/192985/kevin-de-bruyn...</td>
      <td>Belgium</td>
      <td>...</td>
      <td>64</td>
      <td>78</td>
      <td>\n163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Neymar da Silva Santos Jr.</td>
      <td>http://sofifa.com/player/190871/neymar-da-silv...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>36</td>
      <td>59</td>
      <td>\n273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Robert Lewandowski</td>
      <td>http://sofifa.com/player/188545/robert-lewando...</td>
      <td>Poland</td>
      <td>...</td>
      <td>43</td>
      <td>82</td>
      <td>\n182</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kylian Mbappé</td>
      <td>http://sofifa.com/player/231747/kylian-mbappe/...</td>
      <td>France</td>
      <td>...</td>
      <td>39</td>
      <td>76</td>
      <td>\n646</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alisson Ramses Becker</td>
      <td>http://sofifa.com/player/212831/alisson-ramses...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>51</td>
      <td>91</td>
      <td>\n79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mohamed Salah</td>
      <td>http://sofifa.com/player/209331/mohamed-salah/...</td>
      <td>Egypt</td>
      <td>...</td>
      <td>45</td>
      <td>75</td>
      <td>\n164</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sadio Mané</td>
      <td>http://sofifa.com/player/208722/sadio-mane/210...</td>
      <td>Senegal</td>
      <td>...</td>
      <td>44</td>
      <td>76</td>
      <td>\n170</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 76 columns</p>
</div>



## Remove all new line characters in columns

### Find columns with \n characters

Let's see which columns have this character


```python
columns_with_n = list()
for columns in fifa_df.columns:
    #The any() method will return true if any of the values of the column fits the condition
    if fifa_df[columns].astype(str).str.contains('\n').any():
        columns_with_n.append(columns)
print(columns_with_n)
```

    ['Team & Contract', 'Hits']


Looks like there are only two columns in the dataframe that have the character, being 'Team & Contract' and 'Hits'. Now let's get rid of the unwanted character.

### Team and Contract column


```python
team_contract_clean = fifa_df['Team & Contract'].str.replace('\n','')
team_contract_clean.head(10)
```




    0           FC Barcelona2004 ~ 2021
    1               Juventus2018 ~ 2022
    2        Atlético Madrid2014 ~ 2023
    3        Manchester City2015 ~ 2023
    4    Paris Saint-Germain2017 ~ 2022
    5      FC Bayern München2014 ~ 2023
    6    Paris Saint-Germain2018 ~ 2022
    7              Liverpool2018 ~ 2024
    8              Liverpool2017 ~ 2023
    9              Liverpool2016 ~ 2023
    Name: Team & Contract, dtype: object



Now, there are 4 cases in this column:
 - Players on a team, with the format: {team}{year start - year end}
 - Players on loan, with the format: {team}{end of loan}
 - Teams that start with a number, with the format: 1. {team}{year start - year end} or 1. {team}{end of loan}
 - Players that don't belong to any team: {country}{'free'}

#### Creating boolean masks

I will create boolean masks to tell which entries correspond to each case


```python
on_loan_entries = team_contract_clean.str.contains('On Loan')
team_contract_clean.loc[on_loan_entries]
```




    205                  Tottenham HotspurJun 30, 2021 On Loan
    250                Paris Saint-GermainJun 30, 2021 On Loan
    257                             FulhamJun 30, 2021 On Loan
    299                    Atlético MadridJun 30, 2021 On Loan
    305                             NapoliJun 30, 2021 On Loan
                                   ...                        
    18497                     Macarthur FCAug 31, 2021 On Loan
    18569    Shijiazhuang Ever Bright F.C.Dec 31, 2020 On Loan
    18580                        StevenageJun 30, 2021 On Loan
    18638    Shijiazhuang Ever Bright F.C.Dec 31, 2020 On Loan
    18696                 Guangzhou R&F FCDec 31, 2020 On Loan
    Name: Team & Contract, Length: 1013, dtype: object




```python
free_entries = team_contract_clean.str.contains('Free')
team_contract_clean.loc[free_entries]
```




    288                     BrazilFree
    292                     BrazilFree
    369                    EcuadorFree
    370                    UruguayFree
    371                    UruguayFree
                       ...            
    17204     United Arab EmiratesFree
    17303                    IndiaFree
    17668                    IndiaFree
    17670                    IndiaFree
    18208                    IndiaFree
    Name: Team & Contract, Length: 238, dtype: object




```python
number_start_entries = team_contract_clean.str.contains(r'^[0-9]')
team_contract_clean.loc[number_start_entries]
```




    354               1. FC Union Berlin2020 ~ 2023
    819      1. FC Union BerlinJun 30, 2021 On Loan
    1007                      1. FC Köln2019 ~ 2023
    1030                      1. FC Köln2011 ~ 2023
    1125                      1. FC Köln2012 ~ 2023
                              ...                  
    17706           1. FC Kaiserslautern2019 ~ 2022
    17899              1. FC Saarbrücken2019 ~ 2022
    18028          1. FC Heidenheim 18462019 ~ 2021
    18146              1. FC Saarbrücken2018 ~ 2022
    18830           1. FC Kaiserslautern2020 ~ 2022
    Name: Team & Contract, Length: 234, dtype: object



#### Extracting the team for the On Loan entries

For the On Loan entries, as I said before, we have the corresponding team and the end date of the loan, this last piece of information is redundant as we already have a column that stores that data, as it's shown in the next cell


```python
fifa_df_original[['Team & Contract', 'Loan Date End']].loc[on_loan_entries]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Team &amp; Contract</th>
      <th>Loan Date End</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>205</th>
      <td>\n\n\n\nTottenham Hotspur\nJun 30, 2021 On Loa...</td>
      <td>Jun 30, 2021</td>
    </tr>
    <tr>
      <th>250</th>
      <td>\n\n\n\nParis Saint-Germain\nJun 30, 2021 On L...</td>
      <td>Jun 30, 2021</td>
    </tr>
    <tr>
      <th>257</th>
      <td>\n\n\n\nFulham\nJun 30, 2021 On Loan\n\n</td>
      <td>Jun 30, 2021</td>
    </tr>
    <tr>
      <th>299</th>
      <td>\n\n\n\nAtlético Madrid\nJun 30, 2021 On Loan\n\n</td>
      <td>Jun 30, 2021</td>
    </tr>
    <tr>
      <th>305</th>
      <td>\n\n\n\nNapoli\nJun 30, 2021 On Loan\n\n</td>
      <td>Jun 30, 2021</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18497</th>
      <td>\n\n\n\nMacarthur FC\nAug 31, 2021 On Loan\n\n</td>
      <td>Aug 31, 2021</td>
    </tr>
    <tr>
      <th>18569</th>
      <td>\n\n\n\nShijiazhuang Ever Bright F.C.\nDec 31,...</td>
      <td>Dec 31, 2020</td>
    </tr>
    <tr>
      <th>18580</th>
      <td>\n\n\n\nStevenage\nJun 30, 2021 On Loan\n\n</td>
      <td>Jun 30, 2021</td>
    </tr>
    <tr>
      <th>18638</th>
      <td>\n\n\n\nShijiazhuang Ever Bright F.C.\nDec 31,...</td>
      <td>Dec 31, 2020</td>
    </tr>
    <tr>
      <th>18696</th>
      <td>\n\n\n\nGuangzhou R&amp;F FC\nDec 31, 2020 On Loan...</td>
      <td>Dec 31, 2020</td>
    </tr>
  </tbody>
</table>
<p>1013 rows × 2 columns</p>
</div>



Single digit dates don't have a zero as a prefix, so the '{date} On Loan' string is not fixed, so we have to determined which cases fall into this category, as well, we have other instances where the team starts with a number, so we must take care of these as well


```python
#An example of what I mentioned before
team_contract_clean.iloc[2185]
```




    'FC CincinnatiJul 5, 2021 On Loan'




```python
#Extracting the number day 
number_day_on_loan = team_contract_clean.str[-16:-14]
#Getting rid of the whitespaces
number_day_on_loan = number_day_on_loan.str.replace(' ','')
#Extracting the team from the column
team_on_loan = np.where(on_loan_entries,
                       np.where(number_day_on_loan.str.len() == 1, team_contract_clean.str[:-19], team_contract_clean.str[:-20]),  
                       'NA')
#Getting rid of the '1. '
for i in range(len(team_on_loan)):
    if number_start_entries[i] and on_loan_entries[i]:
        team_on_loan[i] = team_on_loan[i][3:]
team_on_loan = pd.Series(team_on_loan)
team_on_loan.loc[on_loan_entries]
```




    205                  Tottenham Hotspur
    250                Paris Saint-Germain
    257                             Fulham
    299                    Atlético Madrid
    305                             Napoli
                         ...              
    18497                     Macarthur FC
    18569    Shijiazhuang Ever Bright F.C.
    18580                        Stevenage
    18638    Shijiazhuang Ever Bright F.C.
    18696                 Guangzhou R&F FC
    Length: 1013, dtype: object



I created a new Series to save all the modified entries and then set this Series as the new team column.


```python
#This is important to pass only the copy of the values, to prevent of modifying the team_contract_clean
team_column = pd.Series(team_contract_clean.values.copy())
#Storing the On Loan modified entries
for i in range(len(team_contract_clean)):
    if on_loan_entries[i]:
        team_column[i] = team_on_loan[i]
    else:
        team_column[i]
team_column.loc[on_loan_entries]
```




    205                  Tottenham Hotspur
    250                Paris Saint-Germain
    257                             Fulham
    299                    Atlético Madrid
    305                             Napoli
                         ...              
    18497                     Macarthur FC
    18569    Shijiazhuang Ever Bright F.C.
    18580                        Stevenage
    18638    Shijiazhuang Ever Bright F.C.
    18696                 Guangzhou R&F FC
    Length: 1013, dtype: object



#### Working on the Free Entries

Now that he have covered all of the On Loan, and, On Loan and Number Start entries, let's move onto the no team entries. I will store these values as 'NA' because there is no associated team to the player.


```python
for i in range(len(team_column)):
    if free_entries[i]:
        team_column[i] = 'NA'
    else:
        team_column[i]
team_column[free_entries]
```




    288      NA
    292      NA
    369      NA
    370      NA
    371      NA
             ..
    17204    NA
    17303    NA
    17668    NA
    17670    NA
    18208    NA
    Length: 238, dtype: object



#### Working on the remaining cases

Now let's keep just the team names


```python
for i in range(len(team_column)):
    if on_loan_entries[i] == True or free_entries[i] == True:
        team_column[i]
    else:
        team_column[i] = team_column[i][:-11]
team_column.head()        
```




    0           FC Barcelona
    1               Juventus
    2        Atlético Madrid
    3        Manchester City
    4    Paris Saint-Germain
    dtype: object




```python
for i in range(len(team_column)):
    if number_start_entries[i] and on_loan_entries[i] == False:
        team_column[i] = team_column[i][3:]
team_column.head()
```




    0           FC Barcelona
    1               Juventus
    2        Atlético Madrid
    3        Manchester City
    4    Paris Saint-Germain
    dtype: object



Now our team column is complete and clean, let's add it into our dataframe

#### Adding Team Column


```python
#Getting the position of the 'Team & Contract' inside the Dataframe 
fifa_df.columns.get_loc('Team & Contract')
```




    8




```python
fifa_df.insert(9,'Team',team_column)
```


```python
fifa_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LongName</th>
      <th>playerUrl</th>
      <th>Nationality</th>
      <th>...</th>
      <th>DEF</th>
      <th>PHY</th>
      <th>Hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lionel Messi</td>
      <td>http://sofifa.com/player/158023/lionel-messi/2...</td>
      <td>Argentina</td>
      <td>...</td>
      <td>38</td>
      <td>65</td>
      <td>\n372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C. Ronaldo dos Santos Aveiro</td>
      <td>http://sofifa.com/player/20801/c-ronaldo-dos-s...</td>
      <td>Portugal</td>
      <td>...</td>
      <td>35</td>
      <td>77</td>
      <td>\n344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jan Oblak</td>
      <td>http://sofifa.com/player/200389/jan-oblak/210005/</td>
      <td>Slovenia</td>
      <td>...</td>
      <td>52</td>
      <td>90</td>
      <td>\n86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kevin De Bruyne</td>
      <td>http://sofifa.com/player/192985/kevin-de-bruyn...</td>
      <td>Belgium</td>
      <td>...</td>
      <td>64</td>
      <td>78</td>
      <td>\n163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Neymar da Silva Santos Jr.</td>
      <td>http://sofifa.com/player/190871/neymar-da-silv...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>36</td>
      <td>59</td>
      <td>\n273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Robert Lewandowski</td>
      <td>http://sofifa.com/player/188545/robert-lewando...</td>
      <td>Poland</td>
      <td>...</td>
      <td>43</td>
      <td>82</td>
      <td>\n182</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kylian Mbappé</td>
      <td>http://sofifa.com/player/231747/kylian-mbappe/...</td>
      <td>France</td>
      <td>...</td>
      <td>39</td>
      <td>76</td>
      <td>\n646</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alisson Ramses Becker</td>
      <td>http://sofifa.com/player/212831/alisson-ramses...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>51</td>
      <td>91</td>
      <td>\n79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mohamed Salah</td>
      <td>http://sofifa.com/player/209331/mohamed-salah/...</td>
      <td>Egypt</td>
      <td>...</td>
      <td>45</td>
      <td>75</td>
      <td>\n164</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sadio Mané</td>
      <td>http://sofifa.com/player/208722/sadio-mane/210...</td>
      <td>Senegal</td>
      <td>...</td>
      <td>44</td>
      <td>76</td>
      <td>\n170</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 77 columns</p>
</div>



#### Contract Column

We know there is a fixed number of characters for entries where the start and end year is stated in our Team & Contract column, and we also know in which columns this is not true (On Loan and Free cases). Let's extract the start and end year respectively and create their own columns.


```python
#fifa_df.drop(columns='Start of Contract', inplace= True)
#fifa_df.drop(columns='End of Contract', inplace= True)
fifa_df.insert(10, 'Start of Contract', None)
fifa_df.insert(11, 'End of Contract', None)
```


```python
fifa_df.loc[(~on_loan_entries) & (~free_entries), 'Start of Contract'] = team_contract_clean.loc[(~on_loan_entries) & (~free_entries)].str[-11:-7]
fifa_df.loc[(~on_loan_entries) & (~free_entries), 'End of Contract'] = team_contract_clean.loc[(~on_loan_entries) & (~free_entries)].str[-4:]    
fifa_df['End of Contract'] = fifa_df['End of Contract'].astype('Int64')
fifa_df['Start of Contract'] = fifa_df['Start of Contract'].astype('Int64')
fifa_df[['Start of Contract', 'End of Contract']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Start of Contract</th>
      <th>End of Contract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2004</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18974</th>
      <td>2020</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>18975</th>
      <td>2020</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>18976</th>
      <td>2018</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>18977</th>
      <td>2020</td>
      <td>2021</td>
    </tr>
    <tr>
      <th>18978</th>
      <td>2020</td>
      <td>2024</td>
    </tr>
  </tbody>
</table>
<p>18979 rows × 2 columns</p>
</div>



It's time to get rid of the Team and Contract column


```python
fifa_df.drop(columns = 'Team & Contract', inplace=True)
```


```python
fifa_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LongName</th>
      <th>playerUrl</th>
      <th>Nationality</th>
      <th>...</th>
      <th>DEF</th>
      <th>PHY</th>
      <th>Hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lionel Messi</td>
      <td>http://sofifa.com/player/158023/lionel-messi/2...</td>
      <td>Argentina</td>
      <td>...</td>
      <td>38</td>
      <td>65</td>
      <td>\n372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C. Ronaldo dos Santos Aveiro</td>
      <td>http://sofifa.com/player/20801/c-ronaldo-dos-s...</td>
      <td>Portugal</td>
      <td>...</td>
      <td>35</td>
      <td>77</td>
      <td>\n344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jan Oblak</td>
      <td>http://sofifa.com/player/200389/jan-oblak/210005/</td>
      <td>Slovenia</td>
      <td>...</td>
      <td>52</td>
      <td>90</td>
      <td>\n86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kevin De Bruyne</td>
      <td>http://sofifa.com/player/192985/kevin-de-bruyn...</td>
      <td>Belgium</td>
      <td>...</td>
      <td>64</td>
      <td>78</td>
      <td>\n163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Neymar da Silva Santos Jr.</td>
      <td>http://sofifa.com/player/190871/neymar-da-silv...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>36</td>
      <td>59</td>
      <td>\n273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Robert Lewandowski</td>
      <td>http://sofifa.com/player/188545/robert-lewando...</td>
      <td>Poland</td>
      <td>...</td>
      <td>43</td>
      <td>82</td>
      <td>\n182</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kylian Mbappé</td>
      <td>http://sofifa.com/player/231747/kylian-mbappe/...</td>
      <td>France</td>
      <td>...</td>
      <td>39</td>
      <td>76</td>
      <td>\n646</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alisson Ramses Becker</td>
      <td>http://sofifa.com/player/212831/alisson-ramses...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>51</td>
      <td>91</td>
      <td>\n79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mohamed Salah</td>
      <td>http://sofifa.com/player/209331/mohamed-salah/...</td>
      <td>Egypt</td>
      <td>...</td>
      <td>45</td>
      <td>75</td>
      <td>\n164</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sadio Mané</td>
      <td>http://sofifa.com/player/208722/sadio-mane/210...</td>
      <td>Senegal</td>
      <td>...</td>
      <td>44</td>
      <td>76</td>
      <td>\n170</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 78 columns</p>
</div>



### Hits column

Let's take a look at the hits column


```python
fifa_df['Hits']
```




    0        \n372
    1        \n344
    2         \n86
    3        \n163
    4        \n273
             ...  
    18974      \n2
    18975      \n3
    18976      \n3
    18977      \n5
    18978      \n2
    Name: Hits, Length: 18979, dtype: object



Looks like the \n character only appears at the start of each entry. But I can see that numbers bigger than 1000 are represented with a K at the end, Let's first get rid of the new line characters


```python
fifa_df['Hits'] = fifa_df['Hits'].str.replace('\n', '')
fifa_df = fifa_df.astype({'Hits': 'str'})
fifa_df['Hits']
```




    0        372
    1        344
    2         86
    3        163
    4        273
            ... 
    18974      2
    18975      3
    18976      3
    18977      5
    18978      2
    Name: Hits, Length: 18979, dtype: object



Let's see which values don't contain only digits


```python
#h_e_w_nd : hits_entries_with_non_digits
h_e_w_nd = fifa_df['Hits'].str.contains(r'[^0-9]')
fifa_df['Hits'].loc[h_e_w_nd]
```




    99      1.3K
    245     1.5K
    279     2.9K
    688       2K
    694       1K
    1273    1.2K
    1399    1.7K
    3673    1.1K
    4045    1.3K
    4064    4.5K
    4117    1.3K
    4680    1.9K
    6203    1.8K
    8126    1.1K
    9707    1.9K
    Name: Hits, dtype: object



All values greater or equal to 1000 are represented with a K. I want to change this in order to convert the column to a nummeric one


```python
fifa_df.loc[h_e_w_nd, 'Hits']
```




    99      1.3K
    245     1.5K
    279     2.9K
    688       2K
    694       1K
    1273    1.2K
    1399    1.7K
    3673    1.1K
    4045    1.3K
    4064    4.5K
    4117    1.3K
    4680    1.9K
    6203    1.8K
    8126    1.1K
    9707    1.9K
    Name: Hits, dtype: object




```python
fifa_df.loc[h_e_w_nd, 'Hits'] = fifa_df.loc[h_e_w_nd, 'Hits'].str[:-1]
fifa_df.loc[h_e_w_nd, 'Hits']
```




    99      1.3
    245     1.5
    279     2.9
    688       2
    694       1
    1273    1.2
    1399    1.7
    3673    1.1
    4045    1.3
    4064    4.5
    4117    1.3
    4680    1.9
    6203    1.8
    8126    1.1
    9707    1.9
    Name: Hits, dtype: object




```python
fifa_df = fifa_df.astype({'Hits': 'float'})
fifa_df.loc[h_e_w_nd, 'Hits'] = fifa_df.loc[h_e_w_nd, 'Hits'] * 1000
fifa_df.loc[h_e_w_nd, 'Hits']
```




    99      1300.0
    245     1500.0
    279     2900.0
    688     2000.0
    694     1000.0
    1273    1200.0
    1399    1700.0
    3673    1100.0
    4045    1300.0
    4064    4500.0
    4117    1300.0
    4680    1900.0
    6203    1800.0
    8126    1100.0
    9707    1900.0
    Name: Hits, dtype: float64




```python
fifa_df = fifa_df.astype({'Hits': 'int'})
fifa_df.loc[h_e_w_nd, 'Hits']
```




    99      1300
    245     1500
    279     2900
    688     2000
    694     1000
    1273    1200
    1399    1700
    3673    1100
    4045    1300
    4064    4500
    4117    1300
    4680    1900
    6203    1800
    8126    1100
    9707    1900
    Name: Hits, dtype: int64



Now our 'Hits' column is a nummeric type, and the entries greater than 1000 are written in number


```python
fifa_df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LongName</th>
      <th>playerUrl</th>
      <th>Nationality</th>
      <th>...</th>
      <th>DEF</th>
      <th>PHY</th>
      <th>Hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lionel Messi</td>
      <td>http://sofifa.com/player/158023/lionel-messi/2...</td>
      <td>Argentina</td>
      <td>...</td>
      <td>38</td>
      <td>65</td>
      <td>372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C. Ronaldo dos Santos Aveiro</td>
      <td>http://sofifa.com/player/20801/c-ronaldo-dos-s...</td>
      <td>Portugal</td>
      <td>...</td>
      <td>35</td>
      <td>77</td>
      <td>344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jan Oblak</td>
      <td>http://sofifa.com/player/200389/jan-oblak/210005/</td>
      <td>Slovenia</td>
      <td>...</td>
      <td>52</td>
      <td>90</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kevin De Bruyne</td>
      <td>http://sofifa.com/player/192985/kevin-de-bruyn...</td>
      <td>Belgium</td>
      <td>...</td>
      <td>64</td>
      <td>78</td>
      <td>163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Neymar da Silva Santos Jr.</td>
      <td>http://sofifa.com/player/190871/neymar-da-silv...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>36</td>
      <td>59</td>
      <td>273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Robert Lewandowski</td>
      <td>http://sofifa.com/player/188545/robert-lewando...</td>
      <td>Poland</td>
      <td>...</td>
      <td>43</td>
      <td>82</td>
      <td>182</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kylian Mbappé</td>
      <td>http://sofifa.com/player/231747/kylian-mbappe/...</td>
      <td>France</td>
      <td>...</td>
      <td>39</td>
      <td>76</td>
      <td>646</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alisson Ramses Becker</td>
      <td>http://sofifa.com/player/212831/alisson-ramses...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>51</td>
      <td>91</td>
      <td>79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mohamed Salah</td>
      <td>http://sofifa.com/player/209331/mohamed-salah/...</td>
      <td>Egypt</td>
      <td>...</td>
      <td>45</td>
      <td>75</td>
      <td>164</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sadio Mané</td>
      <td>http://sofifa.com/player/208722/sadio-mane/210...</td>
      <td>Senegal</td>
      <td>...</td>
      <td>44</td>
      <td>76</td>
      <td>170</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Virgil van Dijk</td>
      <td>http://sofifa.com/player/203376/virgil-van-dij...</td>
      <td>Netherlands</td>
      <td>...</td>
      <td>91</td>
      <td>86</td>
      <td>170</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Marc-André ter Stegen</td>
      <td>http://sofifa.com/player/192448/marc-andre-ter...</td>
      <td>Germany</td>
      <td>...</td>
      <td>45</td>
      <td>88</td>
      <td>93</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Carlos Henrique Venancio Casimiro</td>
      <td>http://sofifa.com/player/200145/carlos-henriqu...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>86</td>
      <td>91</td>
      <td>131</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Thibaut Courtois</td>
      <td>http://sofifa.com/player/192119/thibaut-courto...</td>
      <td>Belgium</td>
      <td>...</td>
      <td>48</td>
      <td>85</td>
      <td>89</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Manuel Neuer</td>
      <td>http://sofifa.com/player/167495/manuel-neuer/2...</td>
      <td>Germany</td>
      <td>...</td>
      <td>57</td>
      <td>86</td>
      <td>90</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Karim Benzema</td>
      <td>http://sofifa.com/player/165153/karim-benzema/...</td>
      <td>France</td>
      <td>...</td>
      <td>40</td>
      <td>76</td>
      <td>169</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sergio Ramos García</td>
      <td>http://sofifa.com/player/155862/sergio-ramos-g...</td>
      <td>Spain</td>
      <td>...</td>
      <td>88</td>
      <td>85</td>
      <td>187</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sergio Agüero</td>
      <td>http://sofifa.com/player/153079/sergio-aguero/...</td>
      <td>Argentina</td>
      <td>...</td>
      <td>33</td>
      <td>73</td>
      <td>103</td>
    </tr>
    <tr>
      <th>18</th>
      <td>N'Golo Kanté</td>
      <td>http://sofifa.com/player/215914/ngolo-kante/21...</td>
      <td>France</td>
      <td>...</td>
      <td>86</td>
      <td>82</td>
      <td>169</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Joshua Kimmich</td>
      <td>http://sofifa.com/player/212622/joshua-kimmich...</td>
      <td>Germany</td>
      <td>...</td>
      <td>81</td>
      <td>79</td>
      <td>317</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 78 columns</p>
</div>



## Players with more than 10 years in a club

Based on the column 'Joined' we are going to see which players have been more than 10 years in a given club. Let's see the data type of the 'Joined' column


```python
fifa_df['Joined']
```




    0         Jul 1, 2004
    1        Jul 10, 2018
    2        Jul 16, 2014
    3        Aug 30, 2015
    4         Aug 3, 2017
                 ...     
    18974     Aug 1, 2020
    18975     Aug 1, 2020
    18976    Jul 13, 2018
    18977     Aug 1, 2020
    18978     Jan 1, 2020
    Name: Joined, Length: 18979, dtype: object



Let's turn this column into a date type, following these steps:
* Extract the month and turn it into a number
* Extracting the day
* Extracting the year
* Put them together and change its type

Note: Keep in mind that we already have the year in the start of contract but we will do an extra step here, just for practice

### Joined as Date Column

#### Month 


```python
month_name = fifa_df['Joined'].str[:3]
month_name
```




    0        Jul
    1        Jul
    2        Jul
    3        Aug
    4        Aug
            ... 
    18974    Aug
    18975    Aug
    18976    Jul
    18977    Aug
    18978    Jan
    Name: Joined, Length: 18979, dtype: object




```python
def month_name_to_number(item):
    month_to_number = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04' , 'May': '05', 'Jun': '06',
               'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    return month_to_number[item]
month_number = month_name.apply(month_name_to_number)
month_number
```




    0        07
    1        07
    2        07
    3        08
    4        08
             ..
    18974    08
    18975    08
    18976    07
    18977    08
    18978    01
    Name: Joined, Length: 18979, dtype: object



#### Day


```python
day_number = fifa_df['Joined'].str[-8:-6]
day_number = day_number.str.replace(' ','')
day_number.loc[day_number.str.len() == 1] = '0' + day_number.loc[day_number.str.len() == 1]
day_number
```




    0        01
    1        10
    2        16
    3        30
    4        03
             ..
    18974    01
    18975    01
    18976    13
    18977    01
    18978    01
    Name: Joined, Length: 18979, dtype: object



#### Year


```python
year = fifa_df['Joined'].str[-4:]
year
```




    0        2004
    1        2018
    2        2014
    3        2015
    4        2017
             ... 
    18974    2020
    18975    2020
    18976    2018
    18977    2020
    18978    2020
    Name: Joined, Length: 18979, dtype: object



#### Putting them together 


```python
joined_date = pd.DataFrame({'day': day_number, 'month': month_number, 'year': year})
joined_date = pd.to_datetime(joined_date)
joined_date.dtypes
```




    dtype('<M8[ns]')



#### Comparing Dates

The dataset is from 2021, by the time I am doing this excersice it is 2024. I am interested in which is the last year in the end of contract column, with this I can know if I would need to use the current date to compare those players who probably are still playing in those teams today.


```python
#Getting the farest in the future year
fifa_df['End of Contract'].loc[fifa_df['End of Contract'].idxmax()]
```




    2028



We know by now that there are values greater than 2024 in the End of Contract column. So we have to take into account those cases. Let's make this dynamic by taking the current year, instead of 2024.


```python
curr_year = pd.Timestamp.now().year
print(curr_year, type(curr_year))
```

    2024 <class 'int'>


So now, I am going to create a Series with the difference of years, the years that a given player has been in a team. For those values in the End of Contract column that are 'NA' we are going to take the difference as zero.


```python
difference = list()
for i in range(len(fifa_df)):
    if pd.isna(fifa_df['End of Contract'][i]):
        difference.append(0)
    elif curr_year >= fifa_df['End of Contract'][i]:
        difference.append(fifa_df['End of Contract'][i] - joined_date[i].year)
    else:
        difference.append(curr_year - joined_date[i].year)
difference_series = pd.Series(difference)
fifa_df.loc[difference_series > 10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LongName</th>
      <th>playerUrl</th>
      <th>Nationality</th>
      <th>...</th>
      <th>DEF</th>
      <th>PHY</th>
      <th>Hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lionel Messi</td>
      <td>http://sofifa.com/player/158023/lionel-messi/2...</td>
      <td>Argentina</td>
      <td>...</td>
      <td>38</td>
      <td>65</td>
      <td>372</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Manuel Neuer</td>
      <td>http://sofifa.com/player/167495/manuel-neuer/2...</td>
      <td>Germany</td>
      <td>...</td>
      <td>57</td>
      <td>86</td>
      <td>90</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Karim Benzema</td>
      <td>http://sofifa.com/player/165153/karim-benzema/...</td>
      <td>France</td>
      <td>...</td>
      <td>40</td>
      <td>76</td>
      <td>169</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sergio Ramos García</td>
      <td>http://sofifa.com/player/155862/sergio-ramos-g...</td>
      <td>Spain</td>
      <td>...</td>
      <td>88</td>
      <td>85</td>
      <td>187</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Harry Kane</td>
      <td>http://sofifa.com/player/202126/harry-kane/210...</td>
      <td>England</td>
      <td>...</td>
      <td>47</td>
      <td>83</td>
      <td>229</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16121</th>
      <td>Kazuyoshi Miura</td>
      <td>http://sofifa.com/player/254704/kazuyoshi-miur...</td>
      <td>Japan</td>
      <td>...</td>
      <td>19</td>
      <td>47</td>
      <td>96</td>
    </tr>
    <tr>
      <th>16435</th>
      <td>Soner Dikmen</td>
      <td>http://sofifa.com/player/252807/soner-dikmen/2...</td>
      <td>Turkey</td>
      <td>...</td>
      <td>54</td>
      <td>57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16532</th>
      <td>John Popelard</td>
      <td>http://sofifa.com/player/248760/john-popelard/...</td>
      <td>France</td>
      <td>...</td>
      <td>49</td>
      <td>59</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16761</th>
      <td>Mohammed Al Wakid</td>
      <td>http://sofifa.com/player/219936/mohammed-al-wa...</td>
      <td>Saudi Arabia</td>
      <td>...</td>
      <td>42</td>
      <td>55</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17052</th>
      <td>Abdullah Al Yousif</td>
      <td>http://sofifa.com/player/235710/abdullah-al-yo...</td>
      <td>Saudi Arabia</td>
      <td>...</td>
      <td>53</td>
      <td>61</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>283 rows × 78 columns</p>
</div>



Let's see how many players have been more than 10 years in their respective teams


```python
difference_bool = np.where(fifa_df['End of Contract'].loc[difference_series > 10], True, False)
difference_bool.sum()
```




    283



So 283 players have been playing more than 10 years in their teams.
Lastly, let's change the Joined column to the date format one.


```python
fifa_df.drop(columns = 'Joined', inplace= True)
fifa_df.insert(18, 'Joined', joined_date)
fifa_df.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LongName</th>
      <th>playerUrl</th>
      <th>Nationality</th>
      <th>...</th>
      <th>DEF</th>
      <th>PHY</th>
      <th>Hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lionel Messi</td>
      <td>http://sofifa.com/player/158023/lionel-messi/2...</td>
      <td>Argentina</td>
      <td>...</td>
      <td>38</td>
      <td>65</td>
      <td>372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C. Ronaldo dos Santos Aveiro</td>
      <td>http://sofifa.com/player/20801/c-ronaldo-dos-s...</td>
      <td>Portugal</td>
      <td>...</td>
      <td>35</td>
      <td>77</td>
      <td>344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jan Oblak</td>
      <td>http://sofifa.com/player/200389/jan-oblak/210005/</td>
      <td>Slovenia</td>
      <td>...</td>
      <td>52</td>
      <td>90</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kevin De Bruyne</td>
      <td>http://sofifa.com/player/192985/kevin-de-bruyn...</td>
      <td>Belgium</td>
      <td>...</td>
      <td>64</td>
      <td>78</td>
      <td>163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Neymar da Silva Santos Jr.</td>
      <td>http://sofifa.com/player/190871/neymar-da-silv...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>36</td>
      <td>59</td>
      <td>273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Robert Lewandowski</td>
      <td>http://sofifa.com/player/188545/robert-lewando...</td>
      <td>Poland</td>
      <td>...</td>
      <td>43</td>
      <td>82</td>
      <td>182</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 78 columns</p>
</div>



## Convert the Value, Wage and the Release Clause into nummeric 

This columns are string, I want to get rid of the currency symbol, and change the data type to nummeric. First, let's get rid of the currency symbol, for that let's see if all entries are in euros.


```python
colname = ['Value', 'Wage', 'Release Clause']
f = lambda x: x.str[0]
sign = fifa_df[colname].apply(f)
print(set(sign['Value']), set(sign['Wage']), set(sign['Release Clause']))
```

    {'€'} {'€'} {'€'}


All the currency signs are for euros, let's get rid of them


```python
f = lambda x: x.str.replace('€','')
fifa_df[colname] = fifa_df[colname].apply(f)
fifa_df[colname].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Wage</th>
      <th>Release Clause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67.5M</td>
      <td>560K</td>
      <td>138.4M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46M</td>
      <td>220K</td>
      <td>75.9M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75M</td>
      <td>125K</td>
      <td>159.4M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>87M</td>
      <td>370K</td>
      <td>161M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>90M</td>
      <td>270K</td>
      <td>166.5M</td>
    </tr>
    <tr>
      <th>5</th>
      <td>80M</td>
      <td>240K</td>
      <td>132M</td>
    </tr>
    <tr>
      <th>6</th>
      <td>105.5M</td>
      <td>160K</td>
      <td>203.1M</td>
    </tr>
    <tr>
      <th>7</th>
      <td>62.5M</td>
      <td>160K</td>
      <td>120.3M</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78M</td>
      <td>250K</td>
      <td>144.3M</td>
    </tr>
    <tr>
      <th>9</th>
      <td>78M</td>
      <td>250K</td>
      <td>144.3M</td>
    </tr>
  </tbody>
</table>
</div>



Now let's see the letters that appear in each column


```python
f = lambda x: x.str[-1]
letter = fifa_df[colname].apply(f)
print(set(letter['Value']), set(letter['Wage']), set(letter['Release Clause']))
```

    {'K', 'M', '0'} {'K', '0'} {'K', 'M', '0'}



```python
def letter_to_number(item):
    if item[-1] == 'M':
        item = item.replace('M','')
        item = int(1000000 * float(item))
        return item
    elif item[-1] == 'K':
        item = item.replace('K','')
        item = int(1000 * float(item))
        return item
    else:
        return int(item)
```


```python
fifa_df['Value'] = fifa_df['Value'].apply(letter_to_number)
fifa_df['Wage'] = fifa_df['Wage'].apply(letter_to_number)
fifa_df['Release Clause'] = fifa_df['Release Clause'].apply(letter_to_number)
```


```python
fifa_df[colname].head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Wage</th>
      <th>Release Clause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67500000</td>
      <td>560000</td>
      <td>138400000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46000000</td>
      <td>220000</td>
      <td>75900000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75000000</td>
      <td>125000</td>
      <td>159400000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>87000000</td>
      <td>370000</td>
      <td>161000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>90000000</td>
      <td>270000</td>
      <td>166500000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>37000000</td>
      <td>150000</td>
      <td>68500000</td>
    </tr>
    <tr>
      <th>96</th>
      <td>35500000</td>
      <td>220000</td>
      <td>72800000</td>
    </tr>
    <tr>
      <th>97</th>
      <td>24000000</td>
      <td>115000</td>
      <td>40800000</td>
    </tr>
    <tr>
      <th>98</th>
      <td>11500000</td>
      <td>93000</td>
      <td>21900000</td>
    </tr>
    <tr>
      <th>99</th>
      <td>45000000</td>
      <td>56000</td>
      <td>85500000</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>




```python
fifa_df[colname].dtypes
```




    Value             int64
    Wage              int64
    Release Clause    int64
    dtype: object



## Columns with Stars to Nummerical 

Some of the columns have a number and a star, this last character is no good for data analysis, so I need to get rid of it. I am going to utilize the piece of code use in the new line characters problem to find which columns have this type of character.


```python
columns_with_star = list()
for columns in fifa_df.columns:
    if fifa_df[columns].astype(str).str.contains('★').any():
        columns_with_star.append(columns)
print(columns_with_star)
```

    ['W/F', 'SM', 'IR']



```python
colname = ['W/F', 'SM', 'IR']
f = lambda x: x.str.replace('★', '')
fifa_df[colname] = fifa_df[colname].apply(f).astype('Int64')
fifa_df[colname].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>W/F</th>
      <th>SM</th>
      <th>IR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
fifa_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LongName</th>
      <th>playerUrl</th>
      <th>Nationality</th>
      <th>...</th>
      <th>DEF</th>
      <th>PHY</th>
      <th>Hits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lionel Messi</td>
      <td>http://sofifa.com/player/158023/lionel-messi/2...</td>
      <td>Argentina</td>
      <td>...</td>
      <td>38</td>
      <td>65</td>
      <td>372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C. Ronaldo dos Santos Aveiro</td>
      <td>http://sofifa.com/player/20801/c-ronaldo-dos-s...</td>
      <td>Portugal</td>
      <td>...</td>
      <td>35</td>
      <td>77</td>
      <td>344</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jan Oblak</td>
      <td>http://sofifa.com/player/200389/jan-oblak/210005/</td>
      <td>Slovenia</td>
      <td>...</td>
      <td>52</td>
      <td>90</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kevin De Bruyne</td>
      <td>http://sofifa.com/player/192985/kevin-de-bruyn...</td>
      <td>Belgium</td>
      <td>...</td>
      <td>64</td>
      <td>78</td>
      <td>163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Neymar da Silva Santos Jr.</td>
      <td>http://sofifa.com/player/190871/neymar-da-silv...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>36</td>
      <td>59</td>
      <td>273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Robert Lewandowski</td>
      <td>http://sofifa.com/player/188545/robert-lewando...</td>
      <td>Poland</td>
      <td>...</td>
      <td>43</td>
      <td>82</td>
      <td>182</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Kylian Mbappé</td>
      <td>http://sofifa.com/player/231747/kylian-mbappe/...</td>
      <td>France</td>
      <td>...</td>
      <td>39</td>
      <td>76</td>
      <td>646</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Alisson Ramses Becker</td>
      <td>http://sofifa.com/player/212831/alisson-ramses...</td>
      <td>Brazil</td>
      <td>...</td>
      <td>51</td>
      <td>91</td>
      <td>79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mohamed Salah</td>
      <td>http://sofifa.com/player/209331/mohamed-salah/...</td>
      <td>Egypt</td>
      <td>...</td>
      <td>45</td>
      <td>75</td>
      <td>164</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Sadio Mané</td>
      <td>http://sofifa.com/player/208722/sadio-mane/210...</td>
      <td>Senegal</td>
      <td>...</td>
      <td>44</td>
      <td>76</td>
      <td>170</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 78 columns</p>
</div>



## Players with High Value but Low Wages

For this problem, we will have to create a scatter plot to compare the relationship between the value of the players and the wages they are earning. With this we will be able to see which players are underpaid.


```python
scatter = fifa_df.plot.scatter('Value', 'Wage')

plt.show();
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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exCAAAQgAAEImBJAhyCATX0l1HY4Xqg4MQYBCEAAAhCAgA8C6BAEsA93Ca8pjhceSyxBAAIQgAAEIOCPADoEAezPY0JqjeOFBBIzEIAABCAAAQj4JoAOQQD7dpowOuB4YVDEBgQgAAEIQAACQQigQxDAQfym5D44XskIMQABCEAAAhCAQEAC6BAEcEDXKa0bjlcaP3pDAAIQgAAEIBCcADokQQG81157yYcffigjR46Uww8/XP785z9nZ3PiiSfKHXfcIQcddJCsW7dOZsyYIQ8//HC32V5++eVy8cUXy/Dhw+W9996TK6+8Ul5//fW8dg0NDXLvvffKGWecIb1795ZXX31VLrnkEvn000/z2o0ZM0ZmzpwpRx11lOzcuVOeffZZmT59urS0tOS1M52bm1vieMFvWnpCAAIQgAAEIFAaAXRIggL4rrvukn/8x3+0xGuuAD7yyCPljTfekKeeekqeeeYZmThxotxyyy1y/vnny6xZs7IzVvF75513yrXXXisrVqyQ8847T0499VQZN26cvP/++9l2L774ohx66KGi7bdv3y633nqr9O/fXxobG7PidsCAAVaftWvXym233SZDhw61RPe8efNk6tSpWVumc/NySxzPixDfQwACEIAABCAQFQF0SEIC+Fvf+pb86U9/skTpo48+mieAX3rpJRk0aJCo2LQvbXPyySdb0eJMJiO9evWSjRs3ymOPPSZXX3211ay2ttaKAq9cuVLOOuss6zMVw2+99ZacdNJJ8vLLL1uf7bfffrJ69WorCqx29brqqqvkxhtvlNGjR8uWLVusz9TGnDlzrCi0Rqr1MpmbibPieCaUaAMBCEAAAhCAQBQE0CEJCeD58+dbQvV3v/udLFq0KCuAVdhqlFZTD+67777s7I4++mgrteGwww6zor3HHHOMvPbaazJ27Fh55513su1UxKqo1oiuXjfffLNceumllqDOvTQNorm5WU455RTrY53D1q1brQiyfelctm3bJtddd50VDTadm4mj4ngmlGgDAQhAAAIQgEAUBNAhCQjg008/XR588EHRnFtNTcgVwBpt/eCDD+SEE04QFcn2NWTIENm8ebOcc845Mnv2bJk2bZo89NBD0qdPn7wcXc3znTt3rhUpXr9+vTz//PMyatQoGT9+fN5KH3jgAZkyZYo1B700mvz444/LNddck9dO0yKW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ufTSS+Wtt96y/md6/exnP5PGxkZZtmyZ/O1vf5Pvfve7cvPNN8uKFStkypQplpkjjzxS3njjDXnqqafkmWeekYkTJ1oC+fzzz5dZs2Zlh1Lxe+edd1qCWvufd955cuqpp8q4cePk/fffz7Z78cUX5dBDDxVtryL71ltvtcS3zsMWtwMGDLD6rF27Vm677TYZOnSoJbrnzZsnU6dOzdoynZsJDxzPhBJtIACBIASiitRGZTfIGukDAQiURgAd4iKAx44dm/nDH/4gO3bskDfffFNUwKoQfvvtt+Xuu++W0aNHW5+Vcp177rny61//WvbZZx/5/PPP5aWXXpJBgwZZQti+Hn30UTn55JNl5MiRoo/vevXqJRs3bpTHHntMrr76aqtZbW2tFQVeuXKlnHXWWdZnKoZVoJ900kny8ssvW5+piF+9erUVBVa7el111VVy4403WuvZsmWL9ZnamDNnjhWF/vDDD63PTOZmygLHMyVFO964xweKEXDzi6hydaOyyw5DAALxE0CHuAjgRYsWZTQd4Mc//rElPFtbW+Xwww+3BPBpp51mRUn333//knZNI8ovvPCCJT43bNhgRWk19eC+++7L2j366KOt1IbDDjvMivYec8wx8tprr8nYsWPlnXfeybZTEauRXo3o6qXRZY1Uq6DOvTQNorm5WU455RTr40WLFsnWrVutCLJ9qcjetm2bXHfdddY69c8mczOFgeOZkqrudrxxX93777R6E78waROEblR2g8yFPhCAQHAC6BAXAbxz586MCt358+dbEdavvvoqK4A1V1Y/79u3r2/6aqtnz55WLvDjjz9u5eOqyNZoq6ZXnHDCCZZt+xoyZIhs3rxZzjnnHJk9e7ZMmzZNHnroIenTp09ejq7m+c6dO9eKFGt+8fPPPy+jRo2S8ePH583xgQcesFIuNO9XL40m6zyuueaavHaaFrF06VIrvcJ0bqYwcDxTUtXbjmhb9e6928r9+EWPQftKQ+PknDrACwJVfyicT1R22XEIQCA+AugQFwG8adOmzEUXXWSJykIBfPbZZ8tdd91lpRT4vf7v//7PEql6aWqCCtddu3bJhAkTZMmSJVb6Q25ucV1dnbS1tVnR3Pvvv9/K+73hhhssAZx7HXvssfLKK69Y+b2aDrFgwQJpb28XfXEt99I83wsvvFA0v1kvjWyrPU3ryL007WPT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kS8/WSF/+8NjvqsdePGISoiEufn2GrrXAV7gm0eY88JWaQS8fNOv9XLwZb9ron3lEECHIIAT8WYcLxHsDOqTQNyPhuMezycOmkPAmAC+bIyKhgkRQIcggBNxPRwvEewM6pNA3C8HxT2eTxw0h4AxAXzZGBUNEyKADolRAJ9xxhly9tlny2GHHSaDBg2S1atXy8MPPyyPPvqoZDJ7Ck6ceOKJcscdd8hBBx0k69atkxkzZljtCq/LL79cLr74Yhk+fLi89957cuWVV8rrr7+e16yhoUHuvfde0bF79+4tr776qlxyySXy6aef5rUbM2aMzJw5U4466ijZuXOnPPvsszJ9+nRpaWnJa2c6Ny9/xvG8CPF9GgjEXR4q7vHSwJg5VCYBfLky97WSVoUOiVEAL126VNauXSv/8R//IRs3bpQf/vCHcs0118h9990nV111lTWTI488Ut544w156qmn5JlnnpGJEyfKLbfcIueff77MmjUrO1sVv3feeadce+21smLFCjnvvPPk1FNPlXHjxsn777+fbffiiy/KoYceKtp++/btcuutt0r//v2lsbExK24HDBhg9dG53XbbbTJ06FBLdM+bN0+mTp2atWU6N5MbBMczoUSbpAnEHcWKe7yk+TJ+5RLAlyt3bytlZeiQGAXwkCFD5Isvvsgb8Ve/+pVMmzZNBg4cKK2trfLSSy9Z0WEVm/alEeKTTz5ZRo4caUWKe/XqZQnoxx57TK6++mqrWW1trRUFXrlypZx11lnWZyqG33rrLTnppJPk5Zdftj7bb7/9rMizRoHVrl4qvm+88UYZPXq0bNmyxfpMbcyZM8eKQn/44YfWZyZzM70xcDxTUrRLkkDceYxxj5ck26TGpipBPOTx5Xg4M0pwAuiQGAVwsaHOOeccefrpp2XEiBHS1NRkRWk19UCjwvZ19NFHW6kNmjqh0d5jjjlGXnvtNRk7dqy888472XYqYjXSqxFdvW6++Wa59NJLLUGde2kaRHNzs5xyyinWx4sWLZKtW7daEWT7UpG9bds2ue6666xosP7ZZG6mrojjmZKiXdIE4n6TPe7xkuYb5/iwjZO2CLzj5c1o/gigQxIWwBqFPf300620g29961vywQcfyAknnCDz5++pK6qR482bN4uK5dmzZ1sR44ceekj69OmTl6Oreb5z5861IsXr16+X559/XkaN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" width="639.9999861283738">


The following piece of code let me plot a scatter plot with an interactive tooltip that shows me the name and index value of a given point when hovering over. I found this piece of code in StackOverflow and I adjusted it to my necessity. 
However, I find the performance of the interactive graph rather slow, so if I am really interested on those players that have a high value, but a low wage, I can narrow down the points to the top players.


```python
%matplotlib notebook
x = fifa_df['Value'].values.copy()
y = fifa_df['Wage'].values.copy()
names = fifa_df['LongName']

fig,ax = plt.subplots()
sc = plt.scatter(x,y,s=50)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->", color = 'w'))
annot.set_visible(False)

def update_annot(ind):
    
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([names[ind['ind'][0]]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.6)
    

def hover(event):
    vis = annot.get_visible()
    cont, ind = sc.contains(event)
    if cont:
        update_annot(ind)
        annot.set_visible(True)
        fig.canvas.draw_idle()
    else:
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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vpnTrlevXnL//ffLeeedJ927d5fXX39drrjiClm3bl1Ou4MPPlgefvhhOemkk6S+vl7mz58v1113nTQ2Nua0M52bl2IRnhchft9KgC/uUUI+Avl10evwU6XueydKIlnliS2dapGGVUtly4L7PNsO/NHVofTrOTANIACB0AjgQyI0wH//938vY8aMkffee0+2b98u3//+9+XWW2+VDz/8UM444ww9k+OOO07eeusteeqpp+Tpp5+WE044QRvkqVOnyuOPP94+W2V+77rrLm2o1f0XX3yxTJ48WY499lj55JNP2tstWLBAjj76aFHtlcm+/fbbtflW88iY2759++p71q5dKzNnzpRBgwZp071w4UK58MIL2/synZuJWhGeCaXKbsMX95W9/12tvpAuWvbWS7KmhyQSnT9+69hfmggwAoNARRPAh0RogPMNddFFF8lvfvMbGTZsmHz11Vfy8ssvS//+/bURzlxz5syRSZMmyUEHHSTpdFpqampk06ZNMnfuXLn22mt1s2QyqaPAK1eulClTpuifKTO8fPlyOeuss+SVV17RPxs+fLh8/vnnOgqs+lXXNddcIzfffLOMHDlStm7dqn+m+nj22Wd1FHrVqlX6ZyZzM32aEJ4pqcpsxxf3lbnvXqs20UW+yg9d9UsOsBdxfg+B8iWAD3FsgH/84x/Lb3/7W20+N27cqKO0KvXgoYceap/ZySefrFMbxo4dq6O9EyZMkDfeeEOOOuooWbFCFXVvvZSJVZFeFdFVl4ouX3nlldpQZ18qDWL37t1y9tln6x8vWbJEduzYoSPImUuZ7J07d8oNN9ygo8Hqn03mZvqoIDxTUpXYji/uK3HXvddsqIt0WneVSCS67JIqEN60aQGBcieAD3FggFW0tlu3bjoXeN68eTof92//9m91tPXTTz+ViRMnyquvvto+s4EDB8rXX38tF1xwgTzzzDMybdo0mTVrlvTo0SMnR1fl+b7wwgs6Uqzyi59//nkZMWKEjB8/PmeVjz76qE65UHm/6lLRZDWPGTNm5LRTaRHLli3T6RWmczN9YBCeKanKa8cX95W35yYrttWFMrnUATYhSxsIVCYBfIgDA/yXv/xFm1R1qdQEZVwbGhrk+OOPl3feeUenP6jUhcxVVVUlzc3NOpr7yCOP6Lzfm266SRvg7OuUU06R1157Tef3qnSIRYsWSUtL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EFYgWADcsWNH06t7zTXXyLp16/z7a7CrM0a0a9dOlixZInXr1pVvvvlGfvvtN5k8ebIcP37cDIu48847pUOHDrJmzRr0wwgQANM8EEAAAQQQQAABCwLBAuBevXrJrFmzQpamd+/e5vMJEyaIbtuoUSMz/MGXVq1aZYLirl27Wjij+MmSADh+6oqSIoAAAggggEAFEggWAHfq1Enmz59vhjLs3LnztLPdunWrGQLx2WefSbVq1eSWW245aZuZM2eaeYF1jmFSaAECYFoHAggggAACCCBgQSBYAFyrVi0zBlinMdP5gkOl6dOnS7du3UwPsM4BrEmnWFu9erXpAe7cubOFM4qfLAmA46euKCkCCCCAAAIIxLlA9erVzWIVmgYNGiS6MpxOd6Zp8eLFsm/fPvP/MWPGyJQpU8w43+LiYhPo6rCGtLQ0OXLkiLRo0UJWrFhh9tHtdAxwv379pGfPnqK9yKwGF76hEADH+ReJ4iOAAAIIIIBA/Aikpqb6F644tdTt27c3Aa2m9PR0EwjrUAYNbjdv3mxmfHjmmWf8cwHrC3E6N7CuJFepUiUz9dm4cePkk08+iR8QSyUlALYET7YIIIAAAggggAB4EaxiAAABE0lEQVQCdgQIgO24kysCCCCAAAIIIICAJQECYEvwZIsAAggggAACCCBgR4AA2I47uSKAAAIIIIAAAghYEiAAtgRPtggggAACCCCAAAJ2BAiA7biTKwIIIIAAAggggIAlAQJgS/BkiwACCCCAAAIIIGBHgADYjju5IoAAAggggAACCFgSIAC2BE+2CCCAAAIIIIAAAnYECIDtuJMrAggggAACCCCAgCUBAmBL8GSLAAIIIIAAAgggYEeAANiOO7kigAACCCCAAAIIWBIgALYET7YIIIAAAggggAACdgQIgO24kysCCCCAAAIIIICAJQECYEvwZIsAAggggAACCCBgR4AA2I47uSKAAAIIIIAAAghYEvg/9H5YdINPf2UAAAAASUVORK5CYII=" width="639.9999861283738">


I will get the mean of the values column, and graph only for those players above the mean.


```python
#Getting the mean of the 'value' column
mean_value = fifa_df['Value'].mean()
number_players_above_mean = len(fifa_df['Value'].loc[fifa_df['Value'] > mean_value])
number_players_above_mean
```




    4127




```python
percentage_above_value_mean = round((number_players_above_mean / len(fifa_df)) * 100, 0)
percentage_above_value_mean
```




    22.0



We got that 22% of the players are above the mean of the value. To improve the graph performance we are only going to plot these values.


```python
%matplotlib notebook
x = fifa_df['Value'].loc[fifa_df['Value'] > mean_value].values.copy()
y = fifa_df['Wage'].loc[fifa_df['Value'] > mean_value].values.copy()
names = fifa_df['LongName'].loc[fifa_df['Value'] > mean_value].values.copy()
indexing = fifa_df.loc[fifa_df['Value'] > mean_value].index

fig,ax = plt.subplots()
sc = plt.scatter(x,y,s=50)

annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->", color='w'))
annot.set_visible(False)

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format("".join(str(indexing[ind["ind"][0]])), 
                           " ".join([names[ind["ind"][0]]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.6)
    

def hover(event):
    vis = annot.get_visible()
    cont, ind = sc.contains(event)
    if cont:
        update_annot(ind)
        annot.set_visible(True)
        fig.canvas.draw_idle()
    else:
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show();
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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KTfccIP8zd/8jfTu3Vs2bNggf/jDH+T222+XXbt2ZWdy9tlny1133SVHHnmkfP311/Lggw/KnDlzOsz06quvliuuuEIGDx4sn376qcyYMUPefPPNvHY9e/aU+++/Xy688ELp1q2bvP7663LllVfK+vXr89odfvjh8vDDD8vJJ58s9fX1snDhQrn++uulsbExr53u3NxUjPDcCFXC7ykxVQm7XNoaO9dIvzOnEwEuDS53Q6CiCeBDQjTA//AP/yCjRo2S9957T7Zv3y4//OEP5bbbbpOPPvpIzjrrLGcmJ554orz11lvy5JNPylNPPSUnnXSSY5CnTp0qjz32WHa2yvzefffdjqFW91966aVy3nnnyQknnCCrVq3Ktlu0aJEce+yxotork33HHXc45lvNI2Nu+/Tp49yzbt06mTVrlgwcONAx3YsXL5aLL74425fu3HSeKISnQ6l821Biqnz31q+VuWlk14eLpP/p6sM2vYscYD1OtIJApRDAh4RogAsNdckll8jvfvc7GTp0qHzzzTfy0ksvSb9+/RwjnLnmzZsnkyZNkkMPPVTS6bR07dpVNm/eLPPnz5frrrvOaZZMJp0o8CeffCJTpkxxfqbM8IoVK+Scc86Rl19+2fnZYYcdJl9++aUTBVb9quvaa6+VW265RYYPHy5bt251fqb6eOaZZ5wo9OrVq52f6cxN98FBeLqkyq8dJabKb0/9XpGuRlJ7d0lVj35UgfB7A+gPAhVAAB9i2QD/5Cc/kRdeeMExn5s2bXKitCr14KGHHsrO7JRTTnFSG8aMGeNEe0899VR544035JhjjpGVK9VJR62XMrEq0qsiuupS0eWrrrrKMdS5l0qD2LNnj5x77rnOj5cuXSo7duxwIsiZS5nsnTt3yo033uhEg9V/68xN95lBeLqkyq0dJabKbUf9X4++Rlr2bJVkbR/qAPu/CfQIgbIngA+xYIBVtLZLly5OLvCCBQucfNy/+7u/c6Ktn332mUycOFFeeeWV7MwGDBggW7ZskYsuukiefvppmT59usyePVu6d++el6Or8nyff/55J1Ks8oufe+45GTZsmIwbNy5vlY8++qiTcqHyftWloslqHjNnzsxrp9Iili9f7qRX6M5N94lBeLqkyqsdJabKaz+DWI2pRra9Nld6jTm39SS4dEoknRZJJCSR8PEkOB/7DYIZfUIA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8sgjTt7vzTff7Bjg3Ou0006T1157zcnvVekQS5YskZaWFlEfruVeKs/38ssvl/79+zs/bmpqcvq755578tq9/fbb8u2338oFF1wgunPrbGt69erl5B5nriFDhsj777+fNevhbKn3UfL/EjQztBkDnDHBTo6kigI6dc46vwrVmfW+gmjemYm4df/uCVLVq78rk1JWkTGcO5YukG6HjZLu/88xkuzavS1n1fkXijO+Mmw73v4P2ffNFzJg0q/yav+aju+2h2pOLfXbZe8XK3LSI/4k/c6cLtV91MEh3mpKZ+ZZinFsv1aTSKq6N/sPigIRb5O+vORR58/dj2iz97QRU83QHgIQCIcABrhzztYM8NixY+W9997LzixjgFXeroreKgN80003SW1tbd7sTz/9dHn11VedvGIVvVUGWBlnlQOce915550ybdo0UZHljAFW/d1777157ZYtWyabNm1yKkhkDLDb3DrDeeuttzopGe2vTLQ6HLl7G8XtNahpr86rcGV+0+miBqcSIscq51Lli9YMGyWJqipTlMbtWw3nNkl2791pOoSzP+mWUGsbt0+PaN61Rap69C2pDnTpxjEfr0kubaGNyY149x5zrtQeMUHrZMD8vFzjLXducHuG3StVeBuXuyAAgegSwABHyADrphlkUiBqampk37592RVEOQUivhFgveiR6SOeMcFuUWDTfuPWPlMxQB2iEebVWVpCZg62//GROQwltXenVPcaUPDUOTdefhjH3DFMoradzS2TmmGS9+2XkSeVwU0x/B4ClUUAAxwhA6z7oRkfwYX3kJrmD4Y3s/IZyS1NoHxWaraSrFlcv0p6jT7T7Oa2cmSZkl7GNxe4wc9nwTTv279UDlIZ/NACfUCgHAhggCNkgNVUVKmxvn37OikHmUsdgqEqNrQvgzZ37tzsh2uqDJoqgaZyf9uXQcv9qE718dVXX3Uog6ZygFUZtG3btjnDTp48WZ599tkOZdDc5qb7UMRFeH5EvTqNhqkUCJc8YF2ecWxny/jaGtfrHqmPybp/93hPWvHPOKrZ+/M2JBPRVR+UupW/C+JjPq/7wH0QgEB5EYiLD7FB3fccYPXRWiYf95e//KWok+F+9atfOWtTZc7q6uqcD+DUQRhPPPGEU/FBHYShDq/o7CAMVblBlUa75JJL5Pzzzy94EIYql5Z7EIYqk1boIIy1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" width="639.9999861283738">


Just by looking at the graph and hovering over the right-most point, we find that Mbappe has a low wage compared to other lower value players.

But let's narrow the points even further by just getting the 0.5% percentile.


```python
percentile_995 = fifa_df['Value'].quantile(.995)
percentile_995
```




    34000000.0




```python
total_players_top = len(fifa_df['Value'].loc[fifa_df['Value'] >= percentile_995])
total_players_top
```




    96



We have found out that the .5% value is 34,000,000 and there are only 96 players that are in this percentile.

So now, I am going to create a dataframe with only these players and their names, wages, and values.


```python
fifa_df_top_percentile_value_players = pd.DataFrame({'Value': fifa_df['Value'].loc[fifa_df['Value'] >= percentile_995].values.copy(),
                                                    'Wage': fifa_df['Wage'].loc[fifa_df['Value'] >= percentile_995].values.copy(),
                                                    'Name': fifa_df['Name'].loc[fifa_df['Value'] >= percentile_995].values.copy()})
fifa_df_top_percentile_value_players
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Wage</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67500000</td>
      <td>560000</td>
      <td>L. Messi</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46000000</td>
      <td>220000</td>
      <td>Cristiano Ronaldo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75000000</td>
      <td>125000</td>
      <td>J. Oblak</td>
    </tr>
    <tr>
      <th>3</th>
      <td>87000000</td>
      <td>370000</td>
      <td>K. De Bruyne</td>
    </tr>
    <tr>
      <th>4</th>
      <td>90000000</td>
      <td>270000</td>
      <td>Neymar Jr</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>91</th>
      <td>35500000</td>
      <td>175000</td>
      <td>O. Dembélé</td>
    </tr>
    <tr>
      <th>92</th>
      <td>34500000</td>
      <td>150000</td>
      <td>Gabriel Jesus</td>
    </tr>
    <tr>
      <th>93</th>
      <td>34500000</td>
      <td>105000</td>
      <td>S. Bergwijn</td>
    </tr>
    <tr>
      <th>94</th>
      <td>36000000</td>
      <td>135000</td>
      <td>M. Ødegaard</td>
    </tr>
    <tr>
      <th>95</th>
      <td>34000000</td>
      <td>105000</td>
      <td>D. Alli</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 3 columns</p>
</div>



I am going to create a Linear Regression object to plot a tendency line to see which players are under the "tended" wage. I managed to do this with the library sklearn, which is use for machine learning.


```python
%matplotlib notebook
x = fifa_df_top_percentile_value_players['Value'].values.copy()
y = fifa_df_top_percentile_value_players['Wage'].values.copy()
indexing = fifa_df.loc[fifa_df['Value'] > mean_value].index

#I created the model
model = LinearRegression()
#In order to pass the values I have to reshape the list of values as a column
x_reshape = x.reshape((-1,1))
#I pass down the values to create the regression line
model.fit(x_reshape,y)
#I turned the values into a Series object
y_values = pd.Series(model.predict(x_reshape))

y_model = y_values.values.copy()
names = fifa_df_top_percentile_value_players['Name'].values.copy()

fig,ax = plt.subplots()
sc = plt.scatter(x,y,s=50)
#I plot the regression line in the figure
plt.plot(x,y_model)


annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->", color='w'))
annot.set_visible(False)

def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format("".join(str(indexing[ind['ind'][0]])), 
                           " ".join([names[ind["ind"][0]]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.6)
    

def hover(event):
    vis = annot.get_visible()
    cont, ind = sc.contains(event)
    if cont:
        update_annot(ind)
        annot.set_visible(True)
        fig.canvas.draw_idle()
    else:
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show();
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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AQhAAAIQgAAZYDLApp4CDLAp8vQLAQhAAAIQgAA+hAywkacA4RnBTqcQgAAEIAABCJABJgNs6inAAJsiT78QgAAEIAABCOBDyAAbeQoQnhHsdAoBCEAAAhCAABng3GaAf/rTn8rTTz/dqNPf/va3Mn369PqfX3TRRXLXXXdJv379ZMuWLfLAAw/IY4891ui+qVOnypVXXildu3aVTz75RK677jp55513UuJat24t9913n4wZM0aKi4tl6dKlctVVV8nmzZtT4vr06SMPPfSQnHvuuVJRUSELFiyQG264QaqqqlLidMdm93RhgO0I8TkEIAABCEAAAn4RwIfkMAOcMMAXXnih7N+/v77nrVu3WkZXXWeffba8++678swzz8izzz4r55xzjtx+++0yceJEmTdvXv09yvzefffdcuONN8pHH30kEyZMkFGjRslZZ50ln376aX3cokWLZMCAAaLiDxw4IHfccYeUlZVJ//79681t27ZtrXs2bdokM2fOlM6dO1ume/HixTJ+/Pj6tnTHpiNWhKdDiRgIQAACEIAABPwggA8xYIA7deoku3fvTtvza6+9Jh06dLCMcOKaM2eOjBw5Unr06CHxeFxatGgh5eXlMnfuXLn++uutsGg0amWB161bJ+PGjbN+pszw6tWr5eKLL5bXX3/d+lnPnj3liy++sLLAql11TZs2TW655Rbp3bt3/bhUG88//7yVhV6/fr0VpzM2XZEiPF1SxEEAAhCAAAQg4DUBfEiADLAytipLq0oPHnzwwfqRnXfeeVZpw+mnn25le4cOHSrLli2T0047TdauXVsfp0ysyvSqjK66brvtNrn66qstQ518qTKIQ4cOySWXXGL9ePny5bJv3z4rg5y41FhUlvqmm26yssG6Y9MVKMLTJUUcBCAAAQhAAAJeE8CHGDDAKnurssCq5OCJJ56Qe+65R2KxmJVt/eyzz2TEiBHyxhtv1I9Mxe7cuVMuu+wyee6552Ty5Mkye/ZsadmyZUqNrqrzffHFF61MsSqrWLhwofTq1UsGDRqUMstHHnlEVBmGqvtVlxrP/PnzU+qQ1c9VWcSqVaus8grdsekKFOHpkiIOAhCAAAQgAAGvCeBDcmiAL7jgAhk4cKBVlqBKGVQGVplZ9YKbKkkYPHiwvPfee1b5g4pJXEVFRVJbW2tlcx9++GGr7vfmm2+2DHDy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kqRQdJ1KKjQhlRIllBBdptFV071GSqnJpbtqCKErjSiTpOkiCb/5fP9znv95Hud5bPuc5yz7nPeaaSacs9dar732Pt+99rr40mlbtmypxJx27UOnnkDpCS99nnDZvt1ikvlaxoW2EuUuTzvh+ikrUKVKFatbt64tWrTItMIV6f8FCIBpDZUuwBikzMS4bO+CCW0l6g2JtsL1Q1uJKsDnMgkQANMuKl2AHyqCmqiNjLZCW6GtRBUgAI4qxX0lqlRxfY4AuLjOd5DacvMhqIna8GgrtBXaSlQBAuCoUtxXokoV1+cIgIvrfAeprTYIGThwoN133332xx9/BCnDrpgpLtufFUwyt1RcaCtR7mG0E66fKO2Ez/yfAAEwLQEBBBBAAAEEEECgqAQIgIvqdFNZBBBAAAEEEEAAAQJg2gACCCCAAAIIIIBAUQkQABfV6aayCCCAAAIIIIAAAgTAtAEEEEAAAQQQQACBohIgAC6q0537ymrL6IkTJ9ppp51mf/31l28ZPWTIkJ3aMrpLly72yiuv2JIlS+zYY4/NfSEDHDGuy+zZs61169bblfioo46yr776KkBNcpdlXBOVoHbt2jZy5Ejr2rWr/7/2tL/33nvtkUceyV0BAxwpjkm9evVs5cqVGUu7adMmq1GjRoCa5DbLOC4qwR577GG33nqrde/e3bQr2urVq23atGl29913J34XrLgm1apVszvvvNMuuugiv3YWL15sQ4cOtXfffTe3Jy3A0Y488ki78cYbrWXLltakSRNbvnx55N+Qiy++2B0OP/xw+/bbb+2OO+6wF198MUAtyDKUAAFwKPkCyHfvvff2oPWHH37wG6y2W9RSZ2+++abfbKMk/VgvXbrUatasaevWrYt884py7FCfycZFAXDVqlX9pp6eFi5caApukpqyMdlzzz3to48+so0bN9q4ceN8a3EFA/phf/DBB5NKYnFNtMV6s2bNStV7t912s5kzZ5raz7nnnptYExU8rou++8QTT5geqIcNG+b3phNPPNHvTQ899JBdf/31iXXJxmTy5MmmYE8mChAvueQSf5A86aSTbMGCBYk1UcE7depkDzzwgH3yySfWsGFD23333SP9hnTr1s2DXT0YzZo1y9tM//79rX379vb2228n2oTCRxcgAI5uxSfLCNx000122223mXqkfv31V//XCy+80J555hlr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FSVFAAEEEEAAgYQL1KxZ0zerUOrXr59pZzgtd6Y0d+5cW7dunf955MiRNnHiRB/nu3XrVg90NayhW7dutnHjRmvatKnNnz/fv6PPaQzwpZdeat27dzf1IrMbXMUNhQA44RcSxUcAAQQQQACB5AjUq1evZOOKsqVu3bq1B7RKPXr08EBYQxkU3H733Xe+4sOIESNK1gLWhDitDayd5KpWrepLn40ePdqmT5+eHJBAJSUADgRPtggggAACCCCAAAJhBAiAw7iTKwIIIIAAAggggEAgAQLgQPBkiwACCCCAAAIIIBBGgAA4jDu5IoAAAggggAACCAQSIAAOBE+2CCCAAAIIIIAAAmEECIDDuJMrAggggAACCCCAQCABAuBA8GSLAAIIIIAAAgggEEaAADiMO7kigAACCCCAAAIIBBIgAA4ET7YIIIAAAggggAACYQQIgMO4kysCCCCAAAIIIIBAIAEC4EDwZIsAAggggAACCCAQRoAAOIw7uSKAAAIIIIAAAggEEiAADgRPtggggAACCCCAAAJhBAiAw7iTKwIIIIAAAggggEAgAQLgQPBkiwACCCCAAAIIIBBGgAA4jDu5IoAAAggggAACCAQS+B/IhGbAHYqnxwAAAABJRU5ErkJggg==" width="639.9999861283738">


We got a positive relationship between the Value and the Wage column which mean that as the Value of a player increases its Wage should do the same.

Thanks to the Linear Regression object I can have the data points that make up the regression line, I will take the difference of the Wage from the dataframe against the Y values of the regression line. This will give me which players differ the most from the trend.


```python
model_difference = fifa_df_top_percentile_value_players['Wage'] - y_values
fifa_df_top_percentile_value_players.insert(3,'Difference', model_difference)
fifa_df_top_percentile_value_players.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Wage</th>
      <th>Name</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67500000</td>
      <td>560000</td>
      <td>L. Messi</td>
      <td>344961.234577</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46000000</td>
      <td>220000</td>
      <td>Cristiano Ronaldo</td>
      <td>66625.952578</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75000000</td>
      <td>125000</td>
      <td>J. Oblak</td>
      <td>-111549.713563</td>
    </tr>
    <tr>
      <th>3</th>
      <td>87000000</td>
      <td>370000</td>
      <td>K. De Bruyne</td>
      <td>99032.769413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>90000000</td>
      <td>270000</td>
      <td>Neymar Jr</td>
      <td>-9571.609843</td>
    </tr>
    <tr>
      <th>5</th>
      <td>80000000</td>
      <td>240000</td>
      <td>R. Lewandowski</td>
      <td>-10890.345657</td>
    </tr>
    <tr>
      <th>6</th>
      <td>105500000</td>
      <td>160000</td>
      <td>K. Mbappé</td>
      <td>-164027.569333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>62500000</td>
      <td>160000</td>
      <td>Alisson</td>
      <td>-40698.133330</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78000000</td>
      <td>250000</td>
      <td>M. Salah</td>
      <td>4845.907181</td>
    </tr>
    <tr>
      <th>9</th>
      <td>78000000</td>
      <td>250000</td>
      <td>S. Mané</td>
      <td>4845.907181</td>
    </tr>
  </tbody>
</table>
</div>



Now I am going to sort the Dataframe by the Difference in ascending order, only taking into account the top 10 players.

We can see that in fact Mbappé was the player with that highest value being underpaid, he is being paid 164,027 euros under what he is "supposed" to be making in accordance to the Value-Wage trend.


```python
fifa_df_top_percentile_value_players.sort_values(by='Difference', ascending = True).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Wage</th>
      <th>Name</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>105500000</td>
      <td>160000</td>
      <td>K. Mbappé</td>
      <td>-164027.569333</td>
    </tr>
    <tr>
      <th>25</th>
      <td>69500000</td>
      <td>82000</td>
      <td>J. Sancho</td>
      <td>-138775.018261</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75000000</td>
      <td>125000</td>
      <td>J. Oblak</td>
      <td>-111549.713563</td>
    </tr>
    <tr>
      <th>52</th>
      <td>41500000</td>
      <td>34000</td>
      <td>G. Donnarumma</td>
      <td>-106467.478538</td>
    </tr>
    <tr>
      <th>84</th>
      <td>35500000</td>
      <td>20000</td>
      <td>Grimaldo</td>
      <td>-103258.720026</td>
    </tr>
    <tr>
      <th>77</th>
      <td>42500000</td>
      <td>47000</td>
      <td>Oyarzabal</td>
      <td>-96335.604956</td>
    </tr>
    <tr>
      <th>74</th>
      <td>45000000</td>
      <td>56000</td>
      <td>E. Haaland</td>
      <td>-94505.921003</td>
    </tr>
    <tr>
      <th>26</th>
      <td>60000000</td>
      <td>110000</td>
      <td>T. Alexander-Arnold</td>
      <td>-83527.817283</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49500000</td>
      <td>80000</td>
      <td>M. de Ligt</td>
      <td>-83412.489887</td>
    </tr>
    <tr>
      <th>47</th>
      <td>57000000</td>
      <td>105000</td>
      <td>K. Havertz</td>
      <td>-79923.438027</td>
    </tr>
  </tbody>
</table>
</div>


