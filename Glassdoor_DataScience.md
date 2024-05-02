# Data Science Job Posts in Glassdoor


```python
#Import the necessary libraries
import numpy as np
import pandas as pd
import re
```

## Importing dataset and creating dataframe


```python
#Dataframe I will be working on
DS_df = pd.read_csv('Uncleaned_DS_jobs.csv')
#Original dataset just for references
DS_original_df = pd.read_csv('Uncleaned_DS_jobs.csv')

DS_df.head(20)
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
      <th>index</th>
      <th>Job Title</th>
      <th>Salary Estimate</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sr Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Description\n\nThe Senior Data Scientist is re...</td>
      <td>3.1</td>
      <td>Healthfirst\n3.1</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1993</td>
      <td>Nonprofit Organization</td>
      <td>Insurance Carriers</td>
      <td>Insurance</td>
      <td>Unknown / Non-Applicable</td>
      <td>EmblemHealth, UnitedHealth Group, Aetna</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Secure our Nation, Ignite your Future\n\nJoin ...</td>
      <td>4.2</td>
      <td>ManTech\n4.2</td>
      <td>Chantilly, VA</td>
      <td>Herndon, VA</td>
      <td>5001 to 10000 employees</td>
      <td>1968</td>
      <td>Company - Public</td>
      <td>Research &amp; Development</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Overview\n\n\nAnalysis Group is one of the lar...</td>
      <td>3.8</td>
      <td>Analysis Group\n3.8</td>
      <td>Boston, MA</td>
      <td>Boston, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1981</td>
      <td>Private Practice / Firm</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>JOB DESCRIPTION:\n\nDo you have a passion for ...</td>
      <td>3.5</td>
      <td>INFICON\n3.5</td>
      <td>Newton, MA</td>
      <td>Bad Ragaz, Switzerland</td>
      <td>501 to 1000 employees</td>
      <td>2000</td>
      <td>Company - Public</td>
      <td>Electrical &amp; Electronic Manufacturing</td>
      <td>Manufacturing</td>
      <td>$100 to $500 million (USD)</td>
      <td>MKS Instruments, Pfeiffer Vacuum, Agilent Tech...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions\n2.9</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>About Us:\n\nHeadquartered in beautiful Santa ...</td>
      <td>4.2</td>
      <td>HG Insights\n4.2</td>
      <td>Santa Barbara, CA</td>
      <td>Santa Barbara, CA</td>
      <td>51 to 200 employees</td>
      <td>2010</td>
      <td>Company - Private</td>
      <td>Computer Hardware &amp; Software</td>
      <td>Information Technology</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>Data Scientist / Machine Learning Expert</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Posting Title\nData Scientist / Machine Learni...</td>
      <td>3.9</td>
      <td>Novartis\n3.9</td>
      <td>Cambridge, MA</td>
      <td>Basel, Switzerland</td>
      <td>10000+ employees</td>
      <td>1996</td>
      <td>Company - Public</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>Biotech &amp; Pharmaceuticals</td>
      <td>$10+ billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Introduction\n\nHave you always wanted to run ...</td>
      <td>3.5</td>
      <td>iRobot\n3.5</td>
      <td>Bedford, MA</td>
      <td>Bedford, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1990</td>
      <td>Company - Public</td>
      <td>Consumer Electronics &amp; Appliances Stores</td>
      <td>Retail</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>Staff Data Scientist - Analytics</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Intuit is seeking a Staff Data Scientist to co...</td>
      <td>4.4</td>
      <td>Intuit - Data\n4.4</td>
      <td>San Diego, CA</td>
      <td>Mountain View, CA</td>
      <td>5001 to 10000 employees</td>
      <td>1983</td>
      <td>Company - Public</td>
      <td>Computer Hardware &amp; Software</td>
      <td>Information Technology</td>
      <td>$2 to $5 billion (USD)</td>
      <td>Square, PayPal, H&amp;R Block</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Ready to write the best chapter of your career...</td>
      <td>3.6</td>
      <td>XSELL Technologies\n3.6</td>
      <td>Chicago, IL</td>
      <td>Chicago, IL</td>
      <td>51 to 200 employees</td>
      <td>2014</td>
      <td>Company - Private</td>
      <td>Enterprise Software &amp; Network Solutions</td>
      <td>Information Technology</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Join our team dedicated to developing and exec...</td>
      <td>4.5</td>
      <td>Novetta\n4.5</td>
      <td>Herndon, VA</td>
      <td>Mc Lean, VA</td>
      <td>501 to 1000 employees</td>
      <td>2012</td>
      <td>Company - Private</td>
      <td>Enterprise Software &amp; Network Solutions</td>
      <td>Information Technology</td>
      <td>$100 to $500 million (USD)</td>
      <td>Leidos, CACI International, Booz Allen Hamilton</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>About Us\n\n\nInterested in working for a huma...</td>
      <td>4.7</td>
      <td>1904labs\n4.7</td>
      <td>Saint Louis, MO</td>
      <td>Saint Louis, MO</td>
      <td>51 to 200 employees</td>
      <td>2016</td>
      <td>Company - Private</td>
      <td>IT Services</td>
      <td>Information Technology</td>
      <td>Unknown / Non-Applicable</td>
      <td>Slalom, Daugherty Business Solutions</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>Data Scientist - Statistics, Early Career</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>*Organization and Job ID**\nJob ID: 310918\n\n...</td>
      <td>3.7</td>
      <td>PNNL\n3.7</td>
      <td>Richland, WA</td>
      <td>Richland, WA</td>
      <td>1001 to 5000 employees</td>
      <td>1965</td>
      <td>Government</td>
      <td>Energy</td>
      <td>Oil, Gas, Energy &amp; Utilities</td>
      <td>$500 million to $1 billion (USD)</td>
      <td>Oak Ridge National Laboratory, National Renewa...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>Data Modeler</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>POSITION PURPOSE:\n\nThe Data Architect/Data M...</td>
      <td>3.1</td>
      <td>Old World Industries\n3.1</td>
      <td>Northbrook, IL</td>
      <td>Northbrook, IL</td>
      <td>201 to 500 employees</td>
      <td>1973</td>
      <td>Company - Private</td>
      <td>Chemical Manufacturing</td>
      <td>Manufacturing</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Position Description:\n\nWant to make a differ...</td>
      <td>3.4</td>
      <td>Mathematica Policy Research\n3.4</td>
      <td>Washington, DC</td>
      <td>Princeton, NJ</td>
      <td>1001 to 5000 employees</td>
      <td>1986</td>
      <td>Company - Private</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>Experienced Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>*******Please Apply using this link: https://a...</td>
      <td>4.4</td>
      <td>Guzman &amp; Griffin Technologies (GGTI)\n4.4</td>
      <td>Washington, DC</td>
      <td>Mays Landing, NJ</td>
      <td>1 to 50 employees</td>
      <td>1997</td>
      <td>Company - Private</td>
      <td>Federal Agencies</td>
      <td>Government</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>Data Scientist - Contract</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>We are an ambitious, well-funded startup with ...</td>
      <td>4.1</td>
      <td>Upside Business Travel\n4.1</td>
      <td>Remote</td>
      <td>Washington, DC</td>
      <td>51 to 200 employees</td>
      <td>2015</td>
      <td>Company - Private</td>
      <td>Internet</td>
      <td>Information Technology</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Job Success Profile\n\nData Scientist\n\nBuckm...</td>
      <td>3.5</td>
      <td>Buckman\n3.5</td>
      <td>Memphis, TN</td>
      <td>Memphis, TN</td>
      <td>1001 to 5000 employees</td>
      <td>1945</td>
      <td>Company - Private</td>
      <td>Chemical Manufacturing</td>
      <td>Manufacturing</td>
      <td>$500 million to $1 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>Data Analyst II</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>The Data Analyst II is responsible for data en...</td>
      <td>4.2</td>
      <td>Insight Enterprises, Inc.\n4.2</td>
      <td>Plano, TX</td>
      <td>Tempe, AZ</td>
      <td>5001 to 10000 employees</td>
      <td>1988</td>
      <td>Company - Public</td>
      <td>Enterprise Software &amp; Network Solutions</td>
      <td>Information Technology</td>
      <td>$5 to $10 billion (USD)</td>
      <td>CDW, PCM, SHI International</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>Medical Lab Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Responsibilities\n\n\nThe Medical Laboratory S...</td>
      <td>3.5</td>
      <td>Tower Health\n3.5</td>
      <td>West Grove, PA</td>
      <td>Reading, PA</td>
      <td>5001 to 10000 employees</td>
      <td>2017</td>
      <td>Nonprofit Organization</td>
      <td>Health Care Services &amp; Hospitals</td>
      <td>Health Care</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>



## Converting the Salary Column into Integers

Looks like the format for the 'Salary Estimate' column is as follows "0-9K - 0-9K (Glassdoor est.)" I want to verify if this true for the whole column


```python
#Create pattern to look for
pattern = re.compile("\\$[0-9].+K-\\$[0-9].+K \\(Glassdoor est\\.\\)")

#Function to apply to the corresponding column
def verify_salary(item):
    if pattern.match(item):
        return True
    else:
        return False

ver_format_salary_mask = DS_df['Salary Estimate'].apply(verify_salary)
DS_df['Salary Estimate'].loc[ver_format_salary_mask == False]
```




    303    $145K-$225K(Employer est.)
    304    $145K-$225K(Employer est.)
    305    $145K-$225K(Employer est.)
    306    $145K-$225K(Employer est.)
    307    $145K-$225K(Employer est.)
    308    $145K-$225K(Employer est.)
    309    $145K-$225K(Employer est.)
    310    $145K-$225K(Employer est.)
    311    $145K-$225K(Employer est.)
    312    $145K-$225K(Employer est.)
    313    $145K-$225K(Employer est.)
    314    $145K-$225K(Employer est.)
    315    $145K-$225K(Employer est.)
    316    $145K-$225K(Employer est.)
    317    $145K-$225K(Employer est.)
    318    $145K-$225K(Employer est.)
    319    $145K-$225K(Employer est.)
    320    $145K-$225K(Employer est.)
    321    $145K-$225K(Employer est.)
    322    $145K-$225K(Employer est.)
    Name: Salary Estimate, dtype: object




```python
ver_format_salary_mask.value_counts()
```




    Salary Estimate
    True     652
    False     20
    Name: count, dtype: int64



I have found that there are 20 entries where the estimate comes from the Employer instead of Glassdoor, and these are the only entries that don't fit the format mentioned above.

So now that I have verify the format for the whole column, I am going to create 3 columns:
 - Min estimate
 - Max estimate
 - Estimation by

### Extracting the min


```python
est_min_sal = DS_df['Salary Estimate'].str.extract(r'([0-9].+)K-')
est_min_sal = est_min_sal.astype(int)
est_min_sal = est_min_sal * 1000
est_min_sal                                                
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
      <td>137000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>137000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>137000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>137000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>137000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>667</th>
      <td>105000</td>
    </tr>
    <tr>
      <th>668</th>
      <td>105000</td>
    </tr>
    <tr>
      <th>669</th>
      <td>105000</td>
    </tr>
    <tr>
      <th>670</th>
      <td>105000</td>
    </tr>
    <tr>
      <th>671</th>
      <td>105000</td>
    </tr>
  </tbody>
</table>
<p>672 rows × 1 columns</p>
</div>



### Extracting the max 


```python
est_max_sal = DS_df['Salary Estimate'].str.extract(r'-.([0-9].+)K')
est_max_sal = est_max_sal.astype(int)
est_max_sal = est_max_sal * 1000
est_max_sal                                                
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
      <td>171000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>171000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>171000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>171000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>171000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>667</th>
      <td>167000</td>
    </tr>
    <tr>
      <th>668</th>
      <td>167000</td>
    </tr>
    <tr>
      <th>669</th>
      <td>167000</td>
    </tr>
    <tr>
      <th>670</th>
      <td>167000</td>
    </tr>
    <tr>
      <th>671</th>
      <td>167000</td>
    </tr>
  </tbody>
</table>
<p>672 rows × 1 columns</p>
</div>



###  Creating columns


```python
DS_df.insert(3, 'Lowest Salary Estimate', est_min_sal)
DS_df.insert(4, 'Highest Salary Estimate', est_max_sal)
DS_df
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
      <th>index</th>
      <th>Job Title</th>
      <th>Salary Estimate</th>
      <th>Lowest Salary Estimate</th>
      <th>Highest Salary Estimate</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sr Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>Description\n\nThe Senior Data Scientist is re...</td>
      <td>3.1</td>
      <td>Healthfirst\n3.1</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1993</td>
      <td>Nonprofit Organization</td>
      <td>Insurance Carriers</td>
      <td>Insurance</td>
      <td>Unknown / Non-Applicable</td>
      <td>EmblemHealth, UnitedHealth Group, Aetna</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>Secure our Nation, Ignite your Future\n\nJoin ...</td>
      <td>4.2</td>
      <td>ManTech\n4.2</td>
      <td>Chantilly, VA</td>
      <td>Herndon, VA</td>
      <td>5001 to 10000 employees</td>
      <td>1968</td>
      <td>Company - Public</td>
      <td>Research &amp; Development</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>Overview\n\n\nAnalysis Group is one of the lar...</td>
      <td>3.8</td>
      <td>Analysis Group\n3.8</td>
      <td>Boston, MA</td>
      <td>Boston, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1981</td>
      <td>Private Practice / Firm</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>JOB DESCRIPTION:\n\nDo you have a passion for ...</td>
      <td>3.5</td>
      <td>INFICON\n3.5</td>
      <td>Newton, MA</td>
      <td>Bad Ragaz, Switzerland</td>
      <td>501 to 1000 employees</td>
      <td>2000</td>
      <td>Company - Public</td>
      <td>Electrical &amp; Electronic Manufacturing</td>
      <td>Manufacturing</td>
      <td>$100 to $500 million (USD)</td>
      <td>MKS Instruments, Pfeiffer Vacuum, Agilent Tech...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions\n2.9</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>667</th>
      <td>667</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>Summary\n\nWe’re looking for a data scientist ...</td>
      <td>3.6</td>
      <td>TRANZACT\n3.6</td>
      <td>Fort Lee, NJ</td>
      <td>Fort Lee, NJ</td>
      <td>1001 to 5000 employees</td>
      <td>1989</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>668</th>
      <td>668</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>Job Description\nBecome a thought leader withi...</td>
      <td>-1.0</td>
      <td>JKGT</td>
      <td>San Francisco, CA</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>669</th>
      <td>669</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>Join a thriving company that is changing the w...</td>
      <td>-1.0</td>
      <td>AccessHope</td>
      <td>Irwindale, CA</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>670</th>
      <td>670</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>100 Remote Opportunity As an AINLP Data Scient...</td>
      <td>5.0</td>
      <td>ChaTeck Incorporated\n5.0</td>
      <td>San Francisco, CA</td>
      <td>Santa Clara, CA</td>
      <td>1 to 50 employees</td>
      <td>-1</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>$1 to $5 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>671</th>
      <td>671</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>Description\n\nThe Data Scientist will be part...</td>
      <td>2.7</td>
      <td>1-800-Flowers\n2.7</td>
      <td>New York, NY</td>
      <td>Carle Place, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1976</td>
      <td>Company - Public</td>
      <td>Wholesale</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>672 rows × 17 columns</p>
</div>




```python
DS_df[['Lowest Salary Estimate', 'Highest Salary Estimate']].dtypes
```




    Lowest Salary Estimate     int64
    Highest Salary Estimate    int64
    dtype: object




```python
estimated_by = np.where(ver_format_salary_mask, 'Glassdoor', 'Employer')
DS_df.insert(5,'Estimated by',estimated_by)
DS_df
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
      <th>index</th>
      <th>Job Title</th>
      <th>Salary Estimate</th>
      <th>Lowest Salary Estimate</th>
      <th>Highest Salary Estimate</th>
      <th>Estimated by</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sr Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Description\n\nThe Senior Data Scientist is re...</td>
      <td>3.1</td>
      <td>Healthfirst\n3.1</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1993</td>
      <td>Nonprofit Organization</td>
      <td>Insurance Carriers</td>
      <td>Insurance</td>
      <td>Unknown / Non-Applicable</td>
      <td>EmblemHealth, UnitedHealth Group, Aetna</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Secure our Nation, Ignite your Future\n\nJoin ...</td>
      <td>4.2</td>
      <td>ManTech\n4.2</td>
      <td>Chantilly, VA</td>
      <td>Herndon, VA</td>
      <td>5001 to 10000 employees</td>
      <td>1968</td>
      <td>Company - Public</td>
      <td>Research &amp; Development</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Overview\n\n\nAnalysis Group is one of the lar...</td>
      <td>3.8</td>
      <td>Analysis Group\n3.8</td>
      <td>Boston, MA</td>
      <td>Boston, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1981</td>
      <td>Private Practice / Firm</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>JOB DESCRIPTION:\n\nDo you have a passion for ...</td>
      <td>3.5</td>
      <td>INFICON\n3.5</td>
      <td>Newton, MA</td>
      <td>Bad Ragaz, Switzerland</td>
      <td>501 to 1000 employees</td>
      <td>2000</td>
      <td>Company - Public</td>
      <td>Electrical &amp; Electronic Manufacturing</td>
      <td>Manufacturing</td>
      <td>$100 to $500 million (USD)</td>
      <td>MKS Instruments, Pfeiffer Vacuum, Agilent Tech...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions\n2.9</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>667</th>
      <td>667</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Summary\n\nWe’re looking for a data scientist ...</td>
      <td>3.6</td>
      <td>TRANZACT\n3.6</td>
      <td>Fort Lee, NJ</td>
      <td>Fort Lee, NJ</td>
      <td>1001 to 5000 employees</td>
      <td>1989</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>668</th>
      <td>668</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Job Description\nBecome a thought leader withi...</td>
      <td>-1.0</td>
      <td>JKGT</td>
      <td>San Francisco, CA</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>669</th>
      <td>669</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Join a thriving company that is changing the w...</td>
      <td>-1.0</td>
      <td>AccessHope</td>
      <td>Irwindale, CA</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>670</th>
      <td>670</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>100 Remote Opportunity As an AINLP Data Scient...</td>
      <td>5.0</td>
      <td>ChaTeck Incorporated\n5.0</td>
      <td>San Francisco, CA</td>
      <td>Santa Clara, CA</td>
      <td>1 to 50 employees</td>
      <td>-1</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>$1 to $5 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>671</th>
      <td>671</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Description\n\nThe Data Scientist will be part...</td>
      <td>2.7</td>
      <td>1-800-Flowers\n2.7</td>
      <td>New York, NY</td>
      <td>Carle Place, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1976</td>
      <td>Company - Public</td>
      <td>Wholesale</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>672 rows × 18 columns</p>
</div>




```python
#Dropping the Salary Estimate column
DS_df.drop(columns='Salary Estimate', inplace= True)
```

## Removing the Numbers from the Company Name


```python
DS_original_df
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
      <th>index</th>
      <th>Job Title</th>
      <th>Salary Estimate</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sr Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Description\n\nThe Senior Data Scientist is re...</td>
      <td>3.1</td>
      <td>Healthfirst\n3.1</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1993</td>
      <td>Nonprofit Organization</td>
      <td>Insurance Carriers</td>
      <td>Insurance</td>
      <td>Unknown / Non-Applicable</td>
      <td>EmblemHealth, UnitedHealth Group, Aetna</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Secure our Nation, Ignite your Future\n\nJoin ...</td>
      <td>4.2</td>
      <td>ManTech\n4.2</td>
      <td>Chantilly, VA</td>
      <td>Herndon, VA</td>
      <td>5001 to 10000 employees</td>
      <td>1968</td>
      <td>Company - Public</td>
      <td>Research &amp; Development</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Overview\n\n\nAnalysis Group is one of the lar...</td>
      <td>3.8</td>
      <td>Analysis Group\n3.8</td>
      <td>Boston, MA</td>
      <td>Boston, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1981</td>
      <td>Private Practice / Firm</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>JOB DESCRIPTION:\n\nDo you have a passion for ...</td>
      <td>3.5</td>
      <td>INFICON\n3.5</td>
      <td>Newton, MA</td>
      <td>Bad Ragaz, Switzerland</td>
      <td>501 to 1000 employees</td>
      <td>2000</td>
      <td>Company - Public</td>
      <td>Electrical &amp; Electronic Manufacturing</td>
      <td>Manufacturing</td>
      <td>$100 to $500 million (USD)</td>
      <td>MKS Instruments, Pfeiffer Vacuum, Agilent Tech...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>$137K-$171K (Glassdoor est.)</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions\n2.9</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>667</th>
      <td>667</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>Summary\n\nWe’re looking for a data scientist ...</td>
      <td>3.6</td>
      <td>TRANZACT\n3.6</td>
      <td>Fort Lee, NJ</td>
      <td>Fort Lee, NJ</td>
      <td>1001 to 5000 employees</td>
      <td>1989</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>668</th>
      <td>668</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>Job Description\nBecome a thought leader withi...</td>
      <td>-1.0</td>
      <td>JKGT</td>
      <td>San Francisco, CA</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>669</th>
      <td>669</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>Join a thriving company that is changing the w...</td>
      <td>-1.0</td>
      <td>AccessHope</td>
      <td>Irwindale, CA</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>670</th>
      <td>670</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>100 Remote Opportunity As an AINLP Data Scient...</td>
      <td>5.0</td>
      <td>ChaTeck Incorporated\n5.0</td>
      <td>San Francisco, CA</td>
      <td>Santa Clara, CA</td>
      <td>1 to 50 employees</td>
      <td>-1</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>$1 to $5 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>671</th>
      <td>671</td>
      <td>Data Scientist</td>
      <td>$105K-$167K (Glassdoor est.)</td>
      <td>Description\n\nThe Data Scientist will be part...</td>
      <td>2.7</td>
      <td>1-800-Flowers\n2.7</td>
      <td>New York, NY</td>
      <td>Carle Place, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1976</td>
      <td>Company - Public</td>
      <td>Wholesale</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>672 rows × 15 columns</p>
</div>



I will find for each of the entries the position of the \n character and afterwards delete the character plus the number


```python
n_character_position_finder = DS_df['Company Name'].str.find('\n')
n_character_position_finder
```




    0      11
    1       7
    2      14
    3       7
    4      18
           ..
    667     8
    668    -1
    669    -1
    670    20
    671    13
    Name: Company Name, Length: 672, dtype: int64




```python
company_name_no_number_ncharacter = []
for i in range(len(n_character_position_finder)):
    if n_character_position_finder[i] == -1:
        company_name_no_number_ncharacter.append(DS_df['Company Name'][i])
    else:
        company_name_no_number_ncharacter.append(DS_df['Company Name'][i][:n_character_position_finder[i]])
company_name_no_number_ncharacter_series = pd.Series(company_name_no_number_ncharacter)
company_name_no_number_ncharacter_series
```




    0               Healthfirst
    1                   ManTech
    2            Analysis Group
    3                   INFICON
    4        Affinity Solutions
                   ...         
    667                TRANZACT
    668                    JKGT
    669              AccessHope
    670    ChaTeck Incorporated
    671           1-800-Flowers
    Length: 672, dtype: object



Let's do a quick check up


```python
check = company_name_no_number_ncharacter_series.str.find('\n')
check.loc[check != -1]
```




    Series([], dtype: int64)



Now that we see there are no more \n characters in the companies' names, let's set this new series as our column


```python
DS_df.columns.get_loc('Company Name')
```




    7




```python
DS_df.drop(columns='Company Name', inplace=True)
DS_df.insert(7,'Company Name', company_name_no_number_ncharacter_series)
DS_df
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
      <th>index</th>
      <th>Job Title</th>
      <th>Lowest Salary Estimate</th>
      <th>Highest Salary Estimate</th>
      <th>Estimated by</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sr Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Description\n\nThe Senior Data Scientist is re...</td>
      <td>3.1</td>
      <td>Healthfirst</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1993</td>
      <td>Nonprofit Organization</td>
      <td>Insurance Carriers</td>
      <td>Insurance</td>
      <td>Unknown / Non-Applicable</td>
      <td>EmblemHealth, UnitedHealth Group, Aetna</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Secure our Nation, Ignite your Future\n\nJoin ...</td>
      <td>4.2</td>
      <td>ManTech</td>
      <td>Chantilly, VA</td>
      <td>Herndon, VA</td>
      <td>5001 to 10000 employees</td>
      <td>1968</td>
      <td>Company - Public</td>
      <td>Research &amp; Development</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Overview\n\n\nAnalysis Group is one of the lar...</td>
      <td>3.8</td>
      <td>Analysis Group</td>
      <td>Boston, MA</td>
      <td>Boston, MA</td>
      <td>1001 to 5000 employees</td>
      <td>1981</td>
      <td>Private Practice / Firm</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>JOB DESCRIPTION:\n\nDo you have a passion for ...</td>
      <td>3.5</td>
      <td>INFICON</td>
      <td>Newton, MA</td>
      <td>Bad Ragaz, Switzerland</td>
      <td>501 to 1000 employees</td>
      <td>2000</td>
      <td>Company - Public</td>
      <td>Electrical &amp; Electronic Manufacturing</td>
      <td>Manufacturing</td>
      <td>$100 to $500 million (USD)</td>
      <td>MKS Instruments, Pfeiffer Vacuum, Agilent Tech...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>667</th>
      <td>667</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Summary\n\nWe’re looking for a data scientist ...</td>
      <td>3.6</td>
      <td>TRANZACT</td>
      <td>Fort Lee, NJ</td>
      <td>Fort Lee, NJ</td>
      <td>1001 to 5000 employees</td>
      <td>1989</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>668</th>
      <td>668</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Job Description\nBecome a thought leader withi...</td>
      <td>-1.0</td>
      <td>JKGT</td>
      <td>San Francisco, CA</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>669</th>
      <td>669</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Join a thriving company that is changing the w...</td>
      <td>-1.0</td>
      <td>AccessHope</td>
      <td>Irwindale, CA</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>670</th>
      <td>670</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>100 Remote Opportunity As an AINLP Data Scient...</td>
      <td>5.0</td>
      <td>ChaTeck Incorporated</td>
      <td>San Francisco, CA</td>
      <td>Santa Clara, CA</td>
      <td>1 to 50 employees</td>
      <td>-1</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>$1 to $5 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>671</th>
      <td>671</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Description\n\nThe Data Scientist will be part...</td>
      <td>2.7</td>
      <td>1-800-Flowers</td>
      <td>New York, NY</td>
      <td>Carle Place, NY</td>
      <td>1001 to 5000 employees</td>
      <td>1976</td>
      <td>Company - Public</td>
      <td>Wholesale</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>672 rows × 17 columns</p>
</div>



## Splitting the Size Column 

I will split this column into the lower and higher range and set the values as integers.


```python
size_clean = DS_df['Size'].str.replace('employees', '')
size_clean
```




    0       1001 to 5000 
    1      5001 to 10000 
    2       1001 to 5000 
    3        501 to 1000 
    4          51 to 200 
                ...      
    667     1001 to 5000 
    668                -1
    669                -1
    670          1 to 50 
    671     1001 to 5000 
    Name: Size, Length: 672, dtype: object



Let's get the lower part of the range


```python
lower_size = size_clean.str.extract(r'([0-9].+)to')
lower_size[0].astype('Int64')
```




    0      1001
    1      5001
    2      1001
    3       501
    4        51
           ... 
    667    1001
    668    <NA>
    669    <NA>
    670       1
    671    1001
    Name: 0, Length: 672, dtype: Int64



Let's get the higher part of the range


```python
higher_size = size_clean.str.extract(r'.+ to ([0-9].+)')
higher_size[0].astype('Int64')
```




    0       5000
    1      10000
    2       5000
    3       1000
    4        200
           ...  
    667     5000
    668     <NA>
    669     <NA>
    670       50
    671     5000
    Name: 0, Length: 672, dtype: Int64



Let's add the series as columns to the dataframe


```python
DS_df.columns.get_loc('Size')
```




    10




```python
DS_df.insert(10, 'Size Lower Limit (Employees)', lower_size)
DS_df.insert(11, 'Size Higher Limit (Employees)', higher_size)
DS_df
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
      <th>index</th>
      <th>Job Title</th>
      <th>Lowest Salary Estimate</th>
      <th>Highest Salary Estimate</th>
      <th>Estimated by</th>
      <th>Job Description</th>
      <th>Rating</th>
      <th>Company Name</th>
      <th>Location</th>
      <th>Headquarters</th>
      <th>Size Lower Limit (Employees)</th>
      <th>Size Higher Limit (Employees)</th>
      <th>Size</th>
      <th>Founded</th>
      <th>Type of ownership</th>
      <th>Industry</th>
      <th>Sector</th>
      <th>Revenue</th>
      <th>Competitors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Sr Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Description\n\nThe Senior Data Scientist is re...</td>
      <td>3.1</td>
      <td>Healthfirst</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>1001</td>
      <td>5000</td>
      <td>1001 to 5000 employees</td>
      <td>1993</td>
      <td>Nonprofit Organization</td>
      <td>Insurance Carriers</td>
      <td>Insurance</td>
      <td>Unknown / Non-Applicable</td>
      <td>EmblemHealth, UnitedHealth Group, Aetna</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Secure our Nation, Ignite your Future\n\nJoin ...</td>
      <td>4.2</td>
      <td>ManTech</td>
      <td>Chantilly, VA</td>
      <td>Herndon, VA</td>
      <td>5001</td>
      <td>10000</td>
      <td>5001 to 10000 employees</td>
      <td>1968</td>
      <td>Company - Public</td>
      <td>Research &amp; Development</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Overview\n\n\nAnalysis Group is one of the lar...</td>
      <td>3.8</td>
      <td>Analysis Group</td>
      <td>Boston, MA</td>
      <td>Boston, MA</td>
      <td>1001</td>
      <td>5000</td>
      <td>1001 to 5000 employees</td>
      <td>1981</td>
      <td>Private Practice / Firm</td>
      <td>Consulting</td>
      <td>Business Services</td>
      <td>$100 to $500 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>JOB DESCRIPTION:\n\nDo you have a passion for ...</td>
      <td>3.5</td>
      <td>INFICON</td>
      <td>Newton, MA</td>
      <td>Bad Ragaz, Switzerland</td>
      <td>501</td>
      <td>1000</td>
      <td>501 to 1000 employees</td>
      <td>2000</td>
      <td>Company - Public</td>
      <td>Electrical &amp; Electronic Manufacturing</td>
      <td>Manufacturing</td>
      <td>$100 to $500 million (USD)</td>
      <td>MKS Instruments, Pfeiffer Vacuum, Agilent Tech...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Data Scientist</td>
      <td>137000</td>
      <td>171000</td>
      <td>Glassdoor</td>
      <td>Data Scientist\nAffinity Solutions / Marketing...</td>
      <td>2.9</td>
      <td>Affinity Solutions</td>
      <td>New York, NY</td>
      <td>New York, NY</td>
      <td>51</td>
      <td>200</td>
      <td>51 to 200 employees</td>
      <td>1998</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>Commerce Signals, Cardlytics, Yodlee</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>667</th>
      <td>667</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Summary\n\nWe’re looking for a data scientist ...</td>
      <td>3.6</td>
      <td>TRANZACT</td>
      <td>Fort Lee, NJ</td>
      <td>Fort Lee, NJ</td>
      <td>1001</td>
      <td>5000</td>
      <td>1001 to 5000 employees</td>
      <td>1989</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>Unknown / Non-Applicable</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>668</th>
      <td>668</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Job Description\nBecome a thought leader withi...</td>
      <td>-1.0</td>
      <td>JKGT</td>
      <td>San Francisco, CA</td>
      <td>-1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>669</th>
      <td>669</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Join a thriving company that is changing the w...</td>
      <td>-1.0</td>
      <td>AccessHope</td>
      <td>Irwindale, CA</td>
      <td>-1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>670</th>
      <td>670</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>100 Remote Opportunity As an AINLP Data Scient...</td>
      <td>5.0</td>
      <td>ChaTeck Incorporated</td>
      <td>San Francisco, CA</td>
      <td>Santa Clara, CA</td>
      <td>1</td>
      <td>50</td>
      <td>1 to 50 employees</td>
      <td>-1</td>
      <td>Company - Private</td>
      <td>Advertising &amp; Marketing</td>
      <td>Business Services</td>
      <td>$1 to $5 million (USD)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>671</th>
      <td>671</td>
      <td>Data Scientist</td>
      <td>105000</td>
      <td>167000</td>
      <td>Glassdoor</td>
      <td>Description\n\nThe Data Scientist will be part...</td>
      <td>2.7</td>
      <td>1-800-Flowers</td>
      <td>New York, NY</td>
      <td>Carle Place, NY</td>
      <td>1001</td>
      <td>5000</td>
      <td>1001 to 5000 employees</td>
      <td>1976</td>
      <td>Company - Public</td>
      <td>Wholesale</td>
      <td>Business Services</td>
      <td>$1 to $2 billion (USD)</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>672 rows × 19 columns</p>
</div>


