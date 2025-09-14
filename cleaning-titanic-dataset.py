#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import environtment
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 220)

# In[2]:


# melakukan load data
df = pd.read_csv('/kaggle/input/titanic-dataset/Titanic Dataset.csv')
df.head()
df.shape
df.columns

# In[3]:


# melakukan EDA singkat
df.info()
df.describe(include='all').T
df.isnull().sum().sort_values(ascending=False)
df.nunique().sort_values()

# In[4]:


# melakukan data cleaning
def clean_titanic(df):
    df = df.copy()  
    df.columns = [c.strip().lower() for c in df.columns]

    # mengisi missing embarked dengan modus
    if 'embarked' in df.columns:
        df['embarked'].fillna(df['embarked'].mode().iloc[0], inplace=True)

    # mengisi missing age dengan median
    if 'age' in df.columns:
        df['age'] = df['age'].fillna(df['age'].median())

    # membuat flag cabin_available
    if 'cabin' in df.columns:
        df['cabin_available'] = df['cabin'].notna().astype(int)
        df.drop(columns=['cabin'], inplace=True, errors='ignore')

    # fitur family size & is_alone
    if set(['sibsp', 'parch']).issubset(df.columns):
        df['family_size'] = df['sibsp'].fillna(0).astype(int) + df['parch'].fillna(0).astype(int) + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)

    # encode sex ke numeric
    if 'sex' in df.columns:
        df['sex_m'] = df['sex'].map({'male': 1, 'female': 0})

    # bucket age
    bins = [0, 12, 18, 35, 60, 200]
    labels = ['child', 'teen', 'young_adult', 'adult', 'senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)

    # bersihkan home.dest (ambil bagian setelah koma terakhir)
    if 'home.dest' in df.columns:
        df['home_dest_clean'] = (
            df['home.dest']
            .fillna('')
            .str.split(',')
            .str[-1]
            .str.strip()
            .replace('', np.nan)
        )

    return df

df_clean = clean_titanic(df)
df_clean.head()
df_clean.isnull().sum()


# In[5]:


# check data quality
def dq_check(df):
    report = {}
    report['rows'] = len(df)
    report['cols'] = df.shape[1]
    report['missing_counts'] = df.isnull().sum().sort_values(ascending=False).head(20)
    report['duplicates'] = df.duplicated().sum()

    # contoh rule: umur wahar
    report['age_minmax'] = (df['age'].min(), df['age'].max()) if 'age' in df.columns else None
    return report

dq_check(df_clean)
    
    
    

# In[6]:


# simpan hasil bersih ke CSV & SQLite (load)
df_clean.to_csv('/kaggle/working/titanic_clean.csv', index=False)

# simpan ke sqlite (agar bisa query sql)
conn = sqlite3.connect('/kaggle/working/titanic.db')
df_clean.to_sql('titanic_clean', conn, if_exists='replace', index=False)

# contoh query via pandas
pd.read_sql_query("SELECT pclass, COUNT(*) as cnt FROM titanic_clean GROUP BY pclass", conn)

# In[7]:


# contoh query sql
q = """
SELECT pclass,
        COUNT(*) AS total,
        SUM(survived) AS survived,
        ROUND(100.0 * SUM(survived) / COUNT(*), 2) AS survival_rate_pct
FROM titanic_clean
GROUP BY pclass
ORDER BY pclass;
"""
pd.read_sql_query(q, conn)

# In[8]:


# contoh via pandas
df_clean.groupby('pclass').agg(total=('survived','size'), survived=('survived','sum')).assign(
    survival_rate_pct=lambda x: 100*x['survived']/x['total']
)

# In[9]:


# visualisasi secara singkat
# survival by sex and pclass
pivot = df_clean.pivot_table(index='pclass', columns='sex', values='survived', aggfunc='mean')
pivot.plot(kind='bar', figsize=(8,5))
plt.title('Survival rate by Pclass & Sex')
plt.ylabel('Survival rate (fraction)')
plt.show()

# In[10]:


# membuat ETL function dan menjalankan reproducible
def run_etl(input_csv, out_csv, out_db):
    df = pd.read_csv(input_csv)
    df_clean = clean_titanic(df)
    
    df_clean.to_csv(out_csv, index=False)
    
    conn = sqlite3.connect(out_db)
    df_clean.to_sql('titanic_clean', conn, if_exists='replace', index=False)
    conn.close()
    
    return df_clean

# run
run_etl('/kaggle/input/titanic-dataset/Titanic Dataset.csv', '/kaggle/working/titanic_clean.csv', '/kaggle/working/titanic.db')

# In[ ]:



