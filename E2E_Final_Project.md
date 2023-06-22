# E2E Final project


```python
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.style as style
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
style.use('fivethirtyeight')

```


```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
```

## Source of data

We have data from Yandex.Realty classified https://realty.yandex.ru containing real estate listings for apartments in St. Petersburg and Leningrad Oblast from 2016 till the middle of August 2018. In this Lab you'll learn how to apply machine learning algorithms to solve business problems. Accurate price prediction can help to find fraudsters automatically and help Yandex.Realty users to make better decisions when buying and selling real estate.

## Statistics


```python
spb_df = pd.read_table('/home/jovyan/__DATA/E2ESML_Spring2023/data/spb.real.estate.archive.2018.tsv')
```

### Calculate price per square meter, get median prices for house and find outliers with the help of this



```python
rent_df = spb_df[spb_df.offer_type == 2]
sell_df = spb_df[spb_df.offer_type == 1]
```


```python
# Create a new column price_per_sq_m and calculate price per sq m
rent_df['price_per_sq_m'] = rent_df.last_price/rent_df.area
```

    /tmp/ipykernel_253/816257644.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      rent_df['price_per_sq_m'] = rent_df.last_price/rent_df.area



```python
# find what's median and mean price per square meter in rent
median_price_per_sq_m = rent_df.price_per_sq_m.median()
mean_price_per_sq_m = rent_df.price_per_sq_m.mean()
print("Median price per sq m in rent: {}".format(median_price_per_sq_m))
print("Mean price per sq m in rent: {}".format(mean_price_per_sq_m))
```

    Median price per sq m in rent: 550.0
    Mean price per sq m in rent: 600.1110692967957



```python
house_rent_df = rent_df.groupby('unified_address').price_per_sq_m.median().reset_index()
```


```python
house_rent_df.rename(columns = {'price_per_sq_m': 'house_price_sqm_median'}, inplace = True)
```


```python
rent_df = rent_df.merge(house_rent_df)
```


```python
outliers = rent_df[(rent_df.price_per_sq_m/rent_df.house_price_sqm_median) > 5]
```


```python
print(len(outliers))
```

    49


### Clean df


```python
rent_df_cleaned = rent_df[~((rent_df.price_per_sq_m/rent_df.house_price_sqm_median) > 5)]
rent_df_cleaned = rent_df_cleaned[rent_df_cleaned.last_price < 1000000]
rent_df_cleaned = rent_df_cleaned[~((rent_df_cleaned.price_per_sq_m > 3000) 
                                     & ((rent_df_cleaned.house_price_sqm_median < 1000) 
                                        | (rent_df_cleaned.house_price_sqm_median == rent_df_cleaned.price_per_sq_m)))]
rent_df_cleaned = rent_df_cleaned[~((rent_df_cleaned.price_per_sq_m < 250) 
                               & (rent_df_cleaned.house_price_sqm_median/rent_df_cleaned.price_per_sq_m >= 2))]
rent_df_cleaned = rent_df_cleaned[~((rent_df_cleaned.price_per_sq_m < 200) 
                                          & (rent_df_cleaned.price_per_sq_m == rent_df_cleaned.house_price_sqm_median))]
```

### Let's see the correlation between variables


```python
sns.heatmap(rent_df_cleaned[['area','kitchen_area','living_area','last_price','rooms']].corr())
```




    <AxesSubplot:>




    
![png](output_19_1.png)
    


### Let's explore the distribution


```python
def visualize_property(df, feature):
    fig, axs = plt.subplots(3, figsize = (8,10))
    #Histogram plot
    axs[0].set_title('Histogram')
    df[feature].hist(ax = axs[0])
    #QQ plot 
    axs[1].set_title('QQ')
    stats.probplot(df[feature], plot=axs[1])
    ##Box plot 
    axs[2].set_title("Box plot")
    sns.boxplot(df[feature], ax = axs[2])
    print("Skewness: %f" % df[feature].skew())
    print("Kurtosis: %f" % df[feature].kurt())




```


```python
visualize_property(rent_df_cleaned, 'last_price')
```

    /opt/conda/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(


    Skewness: 5.281090
    Kurtosis: 52.947956



    
![png](output_22_2.png)
    


As we see, the target variable last_price is not normally distributed. This can reduce the performance of the ML regression models because some assume normal distribution, see sklearn info on preprocessing Therfore we make a log transformation, the resulting distribution looks much better.


```python
import numpy as np
rent_df_cleaned['last_price_log'] = np.log(rent_df_cleaned['last_price'])
visualize_property(rent_df_cleaned, 'last_price_log')
```

    /opt/conda/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(


    Skewness: 1.199456
    Kurtosis: 2.155940



    
![png](output_24_2.png)
    


## Model

The model is Random Forest. The hyperparameters are the following: 
n_estimators=10,
bootstrap=0.8,
max_depth=10,
min_samples_split=3,
max_features=1

## Virtual environment

Create a cloud server (virtual machine)
ssh-keygen -t rsa (key generation)
ssh alisakudrina@158.160.31.35 to transfer calculations to the cloud server

Connect a local server to the cloud server using host and port

Install or update pip (package manager) in a virtual environment (so that we could run another versions of packages later)

- sudo su

- apt-get update

- apt install python3 pip

- pip3 install numpy

- python3 -m pip install --upgrade pip

Load the correct versions for our project

Create a virtual library where to install all the versions we need

- sudo apt install python3.8-venv
- python3 -m venv env
- source env/bin/activate

## Docker

### Doker file

- FROM ubuntu:20.04
- MAINTAINER Kudrina Alisa
- RUN apt-get update -y
- COPY . /opt/gsom_predictor (copy the code inside our image)
- WORKDIR /opt/gsom_predictor (change directory to the path inside our docker container)
- RUN apt install -y python3-pip
- RUN pip3 install -r requirements.txt
- CMD python3 app.py (ending command)

Connect our vm to the git repository. Pull all the previous files from there
create a virtual environment there and install all the necessary libraries
to install all the libraries automatically: pip freeze -> requirements.txt



### How to run an app

- docker build
- docker run
- docker ps
