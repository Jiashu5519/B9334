#!/usr/bin/env /share/share1/share_dff/anaconda3/bin/python

"""
Author: Lira Mota, lmota20@gsb.columbia.edu
Course: Big Data in Finance (Spring 2020)
Date: 2020-02
Code:
    Creates stock_monthly pandas data frame.
    Import CRSP MSE and MSF.

------

Dependence:
fire_pytools

"""

# %% Packages
import wrds
import pandas as pd
import numpy as np
import datetime
from pandas.tseries.offsets import MonthEnd

from import_wrds.crsp_sf import *
from utils.post_event_nan import *

# %% Local Functions

def calculate_cumulative_returns(mdata, tt, min_periods):  # TODO: to be completed
    """
    Calculate past returns for momentum strategy

    Parameters:
    ------------
    mdata: data frame
        crsp monthly data with cols permno, date as index.
    tt: int
        number of periods to cumulate retuns
    min_periods: int
        minimum number of periods. Default tt/2
    here I use the skip = 1 as default
    """
    if min_periods is None:
        min_periods = tt / 2

    start_time = time.time()

    required_cols = ['retadj'] 

    assert set(required_cols).issubset(mdata.columns), "Required columns: {}.".format(', '.join(required_cols))
    
    mdata['retadj'] = mdata['retadj'] + 1
    df = mdata['retadj'].copy()
    df = df.to_frame()
    df = df.reset_index()
    
    # Cumulative Return (adjusted) in 11 months
    
    cret = df.sort_values(['permno', 'date']).groupby('permno').retadj.rolling(window=tt,
                                                                           min_periods=min_periods).apply(np.prod,
                                                                                                          raw=True) - 1
    
    cret = cret.to_frame()
    cret = cret.reset_index()
    cret = cret.groupby('permno').retadj.shift(1)
    
    ## sort_problem
    mdata = mdata.reset_index().sort_values(['permno', 'date'])
    cret = cret.reset_index()
    mdata['cret'] = cret['retadj']
    
    print("Time to calculate %d months past returns: %s seconds" % (tt, str(round(time.time() - start_time, 2))))

    return mdata


def calculate_melag(mdata):
    """
     Parameters:
    ------------
    mdata: data frame
        crsp monthly data with cols permno, date as index and lag_me column

    Notes:
    ------
    If ME is missing, we do not exclude stock, but rather keep it in with last non-missing MElag.
    The stock will be excluded if:
    (i) Delisted;
    (ii) Have a missing ME in the moment of portfolio construction.

    This is different than Ken's method

    EXAMPLE:
    --------
    there seem to be 12 stocks with missing PRC and thus missing ME in July 1926.
    Thus, our number of firms drops from 428 to 416.
    Fama and French report 427 in both July and August, so they also do not seem to exclude these
    rather they probably use the previous MElag for weight and must assume some return in the following month.

    The whole paragraph from the note on Ken French's website:
    ----------------------------------------------------------
    "In May 2015, we revised the method for computing daily portfolio returns
    to match more closely the method for computing monthly portfolio returns.
    Daily files produced before May 2015 drop stocks from a portfolio
    (i) the next time the portfolio is reconstituted, at the end of June, regardless of the CRSP delist date or
    (ii) during any period in which they are missing prices for more than 10 consecutive trading days.
    Daily files produced after May 2015 drop stocks from a portfolio
    (i) immediately after their CRSP delist date or
    (ii) during any period in which they are missing prices for more than 200 consecutive trading days. "
    """

    required_cols = ['edate', 'lag_me', 'lag_dlret']

    set(required_cols).issubset(mdata.columns), "Required columns: {}.".format(', '.join(required_cols))

    df = mdata[required_cols].copy()
    df['melag'] = df.groupby('permno').lag_me.fillna(method='pad')
    df.reset_index(inplace=True)

    # Fill na after delisting
    df = post_event_nan(df=df,
                        event=df.lag_dlret.notnull(),
                        vars=['melag'],
                        id_vars=['permno', 'edate'])

    df.set_index(['permno', 'date'], inplace=True)

    return df[['melag']]


# %% Main Function

def main(save_out=True):

    # %% Set Up
    db = wrds.Connection(wrds_username='jiashu')  # make sure to configure wrds connector before hand.
    DATAPATH = "/Users/sunjs/Desktop/B9334/homeworks/hm_ii/output/" # where to save output?

    start_time = time.time()

    # %% Download CRSP data
    varlist = ['dlret', 'dlretx', 'exchcd', 'naics', 'permco', 'prc', 'ret', 'shrcd', 'shrout', 'siccd', 'ticker']

    start_date = '2000-01-01' # '1970-01-01'
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    freq = 'monthly'  # 'daily'
    permno_list = None  # [10001, 14593, 10107] #
    shrcd_list = None  # [10, 11] #
    exchcd_list = None  # [1, 2, 3] #
    crspm = crsp_sf(varlist,
                    start_date,
                    end_date,
                    freq=freq,
                    permno_list=permno_list,
                    shrcd_list=shrcd_list,
                    exchcd_list=exchcd_list,
                    db=db)

    # %% Create variables

    # Returns adjusted for delisting
    crspm['retadj'] = ((1 + crspm['ret'].fillna(0)) * (1 + crspm['dlret'].fillna(0)) - 1)
    crspm.loc[crspm[['ret', 'dlret']].isnull().all(axis=1), 'retadj'] = np.nan

    # Create Market Equity (ME)
    # SHROUT is the number of publicly held shares, recorded in thousands. ME will be reported in 1,000,000 ($10^6$).
    # If the stock is delisted, we set ME to NaN.
    # Also, some companies have multiple shareclasses (=PERMNOs).
    # To get the company ME, we need to calculate the sum of ME over all shareclasses for one company (=PERMCO).
    # This is used for sorting, but not for weights.
    crspm['me'] = abs(crspm['prc']) * (crspm['shrout'] / 1000)

    # Create MEsum
    crspm['mesum_permco'] = crspm.groupby(['date', 'permco']).me.transform(np.sum, min_count=1)

    # Adjust for delisting
    crspm.loc[crspm.dlret.notnull(), 'me'] = np.nan
    crspm.loc[crspm.dlret.notnull(), 'mesum'] = np.nan

    # Resample data
    # CRSP data has skipping months.
    # Create line to missing  months to facilitate the calculation of lag/past returns

    # -- Check
    crspm.shape
    crspm['days_diff'] = crspm.groupby('permno').date.diff()
    crspm.days_diff.max()

    # -- Resample
    crspm['cpermno'] = 'c' + crspm['permno'].astype(str)
    dcast_crspm = pd.pivot_table(crspm, index='date', columns='cpermno', values='ret').reset_index()

    complete_crsp = pd.wide_to_long(dcast_crspm, stubnames='c', i="date", j='permno')
    complete_crsp.reset_index(inplace=True)

    # -- Fist Date
    fdate = crspm.groupby('permno').date.min().reset_index()
    fdate.rename(columns={'date': 'fdate'}, inplace=True)

    # -- Last Date
    edate = crspm.groupby('permno').date.max().reset_index()
    edate.rename(columns={'date': 'edate'}, inplace=True)

    # -- Date range: select dates within the date range
    drange = pd.merge(fdate, edate)
    complete_crsp = pd.merge(complete_crsp[['date', 'permno']], drange, on=['permno'])
    complete_crsp = complete_crsp[complete_crsp.date >= complete_crsp.fdate]
    complete_crsp = complete_crsp[complete_crsp.date <= complete_crsp.edate]
    complete_crsp.shape

    crspm = pd.merge(complete_crsp[['date', 'permno', 'edate']],
                     crspm,
                     on=['date', 'permno'],
                     how='left')

    crspm.drop(columns='cpermno', inplace=True)

    crspm['days_diff'] = crspm.groupby('permno').date.diff()
    crspm.days_diff.max()
    crspm.days_diff.min()

    # Create MElag
    crspm['lag_me'] = crspm.groupby('permno').me.shift(1)
    crspm['lag_dlret'] = crspm.groupby('permno').dlret.shift(1)

    crspm.sort_values(['permno', 'date'], inplace=True)
    crspm.set_index(['date', 'permno'], inplace=True)
    crspm['melag'] = calculate_melag(crspm)

    # Delete rows that were not in the original data set
    crspm.drop(columns=[x for x in crspm.columns if 'lag_' in x], inplace=True)
    crspm.drop(columns=['edate'], inplace=True)

    crspm = calculate_cumulative_returns(crspm, tt=11, min_periods = 6)

    print("Time to create CRSP monthly: %s seconds" % str(time.time() - start_time))

    # Rankyear
    # Rankyear is the year where we ranked the stock, e.g., for the return of a stock in January 2001,
    # rankyear is 2000, because we ranked it in June 2000
    crspm.reset_index(inplace=True)
    crspm['rankyear'] = crspm.date.dt.year
    crspm.loc[crspm.date.dt.month <= 6, 'rankyear'] = crspm.loc[crspm.date.dt.month <= 6, 'rankyear'] - 1

    if save_out:
        crspm.to_pickle(DATAPATH+'stock_monthly.pkl')
        print("Successfully saved stock_monthly.")
    return crspm


# %% Main
if __name__ == '__main__':
    main()