#!/usr/bin/env /share/share1/share_dff/anaconda3/bin/python

"""
Author: Lira Mota, lmota20@gsb.columbia.edu
Course: Big Data in Finance (Spring 2019)
Date: 2019-02
Code:
    Creates stock_monthly pandas data frame.
    Import CRSP MSE and MSF.

Dependence:
-----------
fire_pytools

Notes:
------
Calculates:
    Book-equity (BE),
    Operating profits (OP) and Operating profitability (OP/BE),
    Investment (INV),
    Earnings,
    Dividends,
    Cash flow and  free cash flow,
    Gross profits
    Leverage

"""

"""
Compustat XpressFeed Variables:
* AT      = Total Assets
* CAPX    = Capital Expenditures
* CEQ     = Common/Ordinary Equity - Total
* COGS    = Cost of Goods Sold
* CSHO    = Common Shares Outstanding
* DLC     = Debt in Current Liabilities
* DLTIS   = Long-Term Debt - Issuance
* DLCCH   = Current Debt - Changes
* DLTT    = Long-Term Debt - Total
* DLTR    = Long-Term Debt - Reduction
* DP      = Depreciation and Amortization
* DV      = Cash Dividends
* DVC     = Divdends common/ordinary
* DVP     = Dividends - Preferred/Preference
* EMP     = Number of Employees
* GP      = Gross Profits
* IB      = Income Before Extraordinary Items
* ICAPT   = Invested Capital - Total
* ITCB    = Investment Tax Credit (Balance Sheet)
* LT      = Total Liabilities
* MIB     = Minority Interest (Balance Sheet)
* NAICS   = North American Industrial Classification System Variable Name
* NAICSH  = North American Industry Classification Codes - Historical Company Variable Name
* NI      = Net Income
* OIADP   = Operating Income After Depreciation
* OIBDP   = Operating Income Before Depreciation
* PPEGT   = "Property, Plant and Equipment - Total (Gross)"
* PRBA    = Postretirement Benefit Assets (from separate pension annual file)
* PRSTKC  = Purchase of Common Stock
* PRSTKCC = Purchase of Common Stock (Cash Flow)
* PSTKRV  = Preferred Stock Redemption Value
* PSTK    = Preferred/Preference Stock (Capital) - Total (kd: = par?)
* PSTKL   = Preferred Stock Liquidating Value
* PSTKRV  = Preferred Stock Liquidating Value
* RE      = Retained Earnings
* REVT    = Revenue - Total
* SALE    = Sales/Turnover Net
* SEQ     = Shareholders Equity
* SIC     = Standard Industrial Classification Code
* SSTK    = Sale of Common and Preferred Stock
* TXDB    = Deferred Taxes Balance Sheet
* TXDI    = Income Taxes - Deferred
* TXDITC  = Deferred Taxes and Investment Tax Credit
* WCAPCH  = Working Capital Change - Total
* XINT    = Interest and Related Expense - Total
* XLR     = Staff Expense - Total
* XRD     = Research and Development Expense
* XSGAQ   = Selling, General and Administrative Expenses (millions)

"""

# %% Packages
import wrds
import pandas as pd
import numpy as np
import datetime
from pandas.tseries.offsets import MonthEnd

# Packages in fire_pytools
from utils.pk_integrity import *
from utils.post_event_nan import *

from import_wrds.compd_fund import *
from import_wrds.compd_aco_pnfnd import *
from import_wrds.crsp_sf import *
from import_wrds.merge_comp_crsp import *
from import_kf.dff_be import *

# %% Functions


def calculate_be(df):
    """
    BE = Share Equity(se) - Prefered Stocks(ps) + Deferred Taxes(dt) - Post retirement Benefit Assets(prba)

    Parameters
    ----------
    df: data frame
        Compustat table with columns ['seq', 'ceq', 'at', 'lt', 'mib', 'pstkrv', 'pstkl', 'pstk', 'txditc', 'txdb', 'itcb', 'prba']

    Definition:
    -----------
    BE is the stockholders book equity, plus balance sheet deferred taxes and investment tax credit (if available),  minus the book value
    of preferred stock. Depending on availability, we use redemption, liquidation, or par value (in that order) for the book value of preferred stock.
    Stockholders equity is the value reported by Moody or COMPUSTAT, if it is available.
    If not, we measure stockholders equity as the book value of common equity plus the par value of preferred stock,
    or the book value of assets minus total liabilities (in that order)".* DFF, JF, 2000, pg 393.
    This is the definition posted on Ken French website:

    """

    required_cols = ['fyear', 'seq', 'ceq', 'at', 'lt', 'mib', 'pstkrv', 'pstkl', 'pstk', 'txditc', 'txdb', 'itcb', 'prba']

    assert set(required_cols).issubset(df.columns), 'Following funda dataitems needed: {}'.format(required_cols)

    df = df[required_cols].copy()

    # Shareholder Equity
    df['se'] = df['seq']

    # Uses Common Equity (ceq) + Preferred Stock (pstk) if SEQ is missing:
    df['se'].fillna((df['ceq'] + df['pstk']), inplace=True)

    # Uses Total Assets (at) - Liabilities (lt) + Minority Interest (mib, if available), if others are missing
    df['se'].fillna((df['at'] - df['lt'] + df['mib'].fillna(0)), inplace=True)

    # Preferred Stock
    # Preferred Stock (Redemption Value)
    df['ps'] = df['pstkrv']
    # Uses Preferred Stock (Liquidating Value (pstkl)) if Preferred Stock (Redemption Value) is missing
    df['ps'].fillna(df['pstkl'], inplace=True)
    # Uses Preferred Stock (Carrying Value (pstk)) if others are missing
    df['ps'].fillna(df['pstk'], inplace=True)

    # Deferred Taxes
    # Uses Deferred Taxes and Investment Tax Credit (txditc)
    df['dt'] = df['txditc']

    # This was Novy-Marx old legacy code. We drop this part to be in accordance with Ken French.
    # Uses Deferred Taxes and Investment Tax Credit(txdb) + Investment Tax Credit (Balance Sheet) (itcb) if txditc is missing

    df['dt'].fillna((df['txdb'].fillna(0) + df['itcb'].fillna(0)), inplace=True)
    # If all measures are missing, set dt to missing
    df.loc[pd.isnull(df['txditc']) & pd.isnull(df['txdb']) & pd.isnull(df['itcb']), 'dt'] = np.nan

    df.loc[df['fyear'] >= 1993, 'dt'] = 0

    # Book Equity
    # Book Equity (BE) = Share Equity (se) - Prefered Stocks (ps) + Deferred Taxes (dt)
    BE = (df['se']  # shareholder equity must be available, otherwise BE is missing
          - df['ps']  # preferred stock must be available, otherwise BE is missing
          + df['dt'].fillna(0)  # add deferred taxes if available
          - df['prba'].fillna(0))  # subtract postretirement benefit assets if available

    return BE


def calculate_op(df):
    """
    Operating Profitability (OP)
    Revenues (SALE (? not sure)) minus cost of goods sold (ITEM 41/COGS), minus selling, general, and administrative expenses (ITEM 189/XSGA),
    minus interest expense (ITEM 15/XINT (? Not Sure)) all divided by book equity.

    Fama, French (2015, JFE, pg.3)
    Ken Frech's website:
    The portfolios for July of year t to June of t+1 include all NYSE, AMEX, and NASDAQ stocks for which we have ME for December of t-1 and June of t,
    (positive) BE for t-1, non-missing revenues data for t-1, and non-missing data for at least one of the following: cost of goods sold, selling,
    general and administrative expenses, or interest expense for t-1.
    """
    required_cols = ['sale', 'cogs', 'xsga', 'xint', 'be']

    assert set(required_cols).issubset(df.columns), 'Following funda dataitems needed: {}'.format(required_cols)

    df = df[required_cols].copy()

    df['cost'] = df[['cogs', 'xsga', 'xint']].sum(axis=1, skipna=True)
    df.loc[df[['cogs', 'xsga', 'xint']].isnull().all(axis=1), 'cost'] = np.nan

    df['op'] = df['sale'] - df['cost']
    df.loc[(df.be > 0), 'opbe'] = df['op'] / df['be']

    return df['op'], df['opbe']


def calculate_inv(df):
    """
    Investment (INV)
    The change in total assets from the fiscal year ending in year t-2 to the fiscal year ending in t-1, divided by t-2 total assets.
    Fama, French (2015, JFE, pg.3)

    Notes:
    ------
    Ken Frech's website:
    The portfolios for July of year t to June of t+1 include all NYSE, AMEX, and NASDAQ stocks for which we have market equity data for June of t and total assets
    data for t-2 and t-1.
    """

    required_cols = ['datadate', 'gvkey', 'fyear', 'at']

    assert set(required_cols).issubset(df.columns), 'Following funda dataitems needed: {}'.format(required_cols)

    df = df[required_cols].copy()

    if any(df.fyear.isnull() & df['at'].notnull()):
        warnings.warn('''Missing fyear with valid at value. Row was arbitrarily deleted. ''')

    # Notice that [gvkey, fyear] is not a primary key for compustat.
    # There are some cases in which there are 2 datadate for the same fyear.
    # In many cases there is one of them have missing.  In this case we keep the entry that is not missing.
    df = df[df.fyear.notnull()]
    df = df[df['at'] > 0]

    pk_integrity(df, ['gvkey', 'fyear'])

    df.sort_values(['gvkey', 'fyear'], inplace=True)
    df['lag_at'] = df.groupby('gvkey').at.shift(1)
    df['inv'] = (df['at'] - df['lag_at']) / df['lag_at']

    # Take care if there are years missing
    df['fdiff'] = df.groupby('gvkey').fyear.diff()
    df.loc[df.fdiff > 1, 'inv'] = np.nan

    # df[df['gvkey'] == '018073']

    return df['inv']

# %% Main Function


def main(save_out=True):

    print("Stock annual calculation.")
    # %% Set Up
    db = wrds.Connection(wrds_username='jiashu')  # make sure to configure wrds connector before hand.
    DATAPATH = "/Users/sunjs/Desktop/B9334/homeworks/hm_ii/output/" # where to save output?

    start_time = time.time()

    # %% Import Data

    varlist = ['conm', 'fyear', 'fyr', 'at', 'capx', 'ceq', 'cogs', 'dlc', 'ib', 'icapt', 'itcb', 'lt', 'mib',
               'naicsh', 'pstk', 'pstkl', 'pstkrv',  'sale', 'seq', 'sich', 'sstk', 'txdb', 'txdi', 'txditc',
               'xint', 'xsga']
    start_date = '1950-01-01'  # '2017-01-01'#
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    freq = 'annual'

    # Download firms fundamentals from Compustat.
    comp_data = compd_fund(varlist=varlist, start_date=start_date, end_date=end_date, freq='annual', db=db)

    del varlist

    # Download pension data
    varlist_aco = ['prba']
    pension_data = compd_aco_pnfnd(varlist=varlist_aco, start_date=start_date, end_date=end_date, freq=freq, db=db)

    del varlist_aco

    # Download Davis, Fama, French BE Data
    dff = dff_be()

    # CRSP ME Data
    varlist_crsp = varlist = ['exchcd', 'naics', 'permco', 'prc', 'shrcd', 'shrout', 'siccd', 'ticker']
    start_date = '1925-01-01'  # '2017-01-01' #
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    freq = 'monthly'  # 'daily'
    crspm = crsp_sf(varlist_crsp,
                    start_date,
                    end_date,
                    freq=freq,
                    db=db)

    del varlist_crsp

    # Merge
    comp = pd.merge(comp_data, pension_data, on=['gvkey', 'datadate'], how='left')

    del (comp_data, pension_data)

    # %% Add variables

    # Add BE
    comp['be'] = calculate_be(comp)

    # Add OP: naming here is to be consistent with risk and returns projects.
    comp['op'], comp['opbe'] = calculate_op(comp)

    # Add INV
    comp['inv_gvkey'] = calculate_inv(comp)
    # comp.loc[comp.gvkey == '006557', ['gvkey', 'datadate', 'fyear', 'inv']]
    # comp.loc[comp.gvkey == '001414', ['gvkey','datadate','fyear','inv','at']]

    # There are 2 entries in which fyear is missing. All variables are null in these cases.
    # comp.loc[comp['fyear'].isnull(), ['gvkey', 'cusip', 'permno', 'fyear', 'at', 'be']]
    comp.dropna(subset=['fyear'], inplace=True)

    # Merged data set has [permno, datadate] as primary key. We need ['permno', 'fyear'] as primary key.
    # Six duplicated cases in which we choose the latest observation.
    # comp[comp[['permno', 'fyear']].duplicated(keep=False)][['permno', 'gvkey', 'datadate', 'fyear', 'at']]
    # comp[comp['permno']==22074][['permno', 'gvkey', 'datadate', 'fyear', 'at']]

    # These are cases where the gvkey changes (probably merger) and the fiscal-year-end (fyr) also changes
    # by keeping last, we put the merger investment into the year it happens
    # That seems reasonable
    comp = comp[~comp.duplicated(subset=['gvkey', 'fyear'], keep='last')]

    # Add rankyear
    comp['rankyear'] = comp['fyear'] + 1

    # %% Add CRSP Variables

    # Link CRSP/Compustat table
    lcomp = merge_compustat_crsp(comp, db=db)
    # Link CRSP/Compustat table
    lcomp = merge_compustat_crsp(comp, db=db)
    print("CRSP and Compsuat merge created %d (fyear, permno) duplicates." %lcomp[lcomp.duplicated(subset=['permno', 'fyear'])].shape[0])
    print("Keeping only the last available datadate per PERMNO.")

    lcomp.sort_values(by=['permno', 'fyear', 'datadate'])
    lcomp = lcomp[~lcomp.duplicated(subset=['permno', 'fyear'], keep='last')]

    #  Calculate INV by PERMCO
    lcomp.sort_values(['permno', 'fyear'], inplace=True)
    lcomp['lag_at'] = lcomp.groupby('permno').at.shift(1)
    lcomp['inv'] = (lcomp['at'] - lcomp['lag_at']) / lcomp['lag_at']
    # Take care if there are years missing
    lcomp['fdiff'] = lcomp.groupby('permno').fyear.diff()
    lcomp.loc[lcomp.fdiff > 1, 'inv'] = np.nan
    lcomp.loc[(lcomp['at'] <= 0), 'inv'] = np.nan
    lcomp.loc[(lcomp['lag_at'] <= 0), 'inv'] = np.nan
    lcomp.drop(columns=['lag_at', 'fdiff'], inplace=True)

    # Add Davis data
    dff.rename(columns={'be': 'be_dff'}, inplace=True)
    # Add PERMCO to DFF data: Important for ME sum later
    pp_key = crspm[['permno', 'permco']].drop_duplicates()
    # max(pp_key.groupby(['permno']).permco.count())
    dff = pd.merge(dff, pp_key, on=['permno'], how='left')

    print('There are %d PERMNOs without a valid PERMCO: not present in the crspm table.' % dff[dff.permco.isnull()].permno.unique())

    lcomp = pd.merge(lcomp, dff, on=['permno', 'permco', 'rankyear'], how='outer')
    lcomp.be.fillna(lcomp['be_dff'], inplace=True)
    lcomp.drop(columns=['be_dff'], inplace=True)
    lcomp.fyear.fillna(lcomp['rankyear'] - 1, inplace=True)

    print('Number of not valid PERMCOs lcomp: %d' % round(lcomp.permco.isnull().sum() / lcomp.shape[0], 4))

    del dff, pp_key
    ## Notice that, since int does not support null, outer merge changes dtype of permco

    lcomp.sort_values(['permno', 'rankyear'], inplace=True)

    # ME, SICCD, TICKER, exchcd and shrcrd from June
    crspm['me'] = crspm.prc.abs()*crspm.shrout

    crspjune = crspm.loc[crspm.date.dt.month == 6, ['permno', 'permco', 'date', 'me', 'exchcd', 'shrcd', 'ticker', 'siccd']]
    crspjune['rankyear'] = crspjune.date.dt.year
    crspjune.drop('date', axis=1, inplace=True)
    stock_annual = lcomp.merge(crspjune, how='outer', on=['permno', 'permco', 'rankyear'])
    stock_annual.sort_values(['permno', 'rankyear'], inplace=True)

    # For summing size over issues of the same firm:
    # we rely on gvkey first
    # and if there is no gvkey (e.g. before lcompustat sample period) we use PERMCO
    stock_annual['gvkey_permco'] = stock_annual.gvkey.fillna(stock_annual['permco'])
    stock_annual['mesum_june'] = stock_annual.groupby(['fyear', 'gvkey_permco'])['me'].transform(np.sum, min_count=1)
    stock_annual.rename(columns={'me': 'mejune'}, inplace=True)
    del crspjune

    # Calculate ME december
    crspdec = crspm.loc[crspm.date.dt.month == 12, ['permno', 'date', 'me']]
    crspdec['fyear'] = crspdec.date.dt.year
    crspdec.drop('date', axis=1, inplace=True)
    crspdec.rename(columns={'me': 'medec'}, inplace=True)
    stock_annual = pd.merge(stock_annual, crspdec, how='left', on=['permno', 'fyear'])
    stock_annual['mesum_dec'] = stock_annual.groupby(['fyear', 'gvkey_permco'])['medec'].transform(np.sum, min_count=1)
    del crspdec, lcomp

    # Calculate book-to-market
    # In accordance with FF1993 and DFF2000 the BEME used to form portfolios in June of year t,
    # is BE for the fiscal year ending in t-1, divided by ME at December of t-1. ME for December
    stock_annual.loc[(stock_annual.be > 0), 'beme'] = stock_annual['be'] / stock_annual['mesum_dec']
    stock_annual.loc[(stock_annual.mesum_dec == 0) | (stock_annual.mesum_dec.isnull()), 'beme'] = np.nan

    # Calculate again variables that depend on be values. Need to consider the DFF data OP
    stock_annual.loc[(stock_annual.be > 0), 'opbe'] = stock_annual['op'] / stock_annual['be']

    # Add CRSP SIC when missing
    stock_annual.sich.fillna(stock_annual['siccd'], inplace=True)

    # Back fill SIC code
    stock_annual['sich_filled'] = stock_annual.sich.copy()
    stock_annual['sich_filled'] = stock_annual.groupby('permno').sich_filled.fillna(method='bfill')
    stock_annual['sich_filled'] = stock_annual.groupby('permno').sich_filled.fillna(method='ffill')

    print('Number of entries with valis sich:')
    print(round(pd.isnull(stock_annual.sich).sum() / stock_annual.shape[0], 4))
    print('Number of entries with valis sich_filled:')
    print(round(pd.isnull(stock_annual.sich_filled).sum() / stock_annual.shape[0], 4))

    pk_integrity(stock_annual, ['permno', 'rankyear'])
    stock_annual.sort_values(['permno', 'rankyear'], inplace=True)

    stock_annual.drop(columns=['gvkey_permco'], inplace=True)

    print("Time to create stock_annual: %s seconds" % str(time.time() - start_time))

    if save_out:
        stock_annual.to_pickle(DATAPATH + 'stock_annual.pkl')
        print("Successfully saved stock_annual.")

    return stock_annual

# %% Main


if __name__ == '__main__':
    main()
