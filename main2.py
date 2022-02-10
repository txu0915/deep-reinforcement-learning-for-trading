from utils import *
import itertools

# from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import os
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


df = YahooDownloader(start_date = '2009-01-01',
                     end_date = '2021-01-01',
                     ticker_list = config.DOW_30_TICKER).fetch_data()

df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
df.sort_values(['date','tic'],ignore_index=True).head()

url = 'https://raw.githubusercontent.com/mariko-sawada/FinRL_with_fundamental_data/main/dow_30_fundamental_wrds.csv'

fund = pd.read_csv(url)

items = [
    'datadate', # Date
    'tic', # Ticker
    'oiadpq', # Quarterly operating income
    'revtq', # Quartely revenue
    'niq', # Quartely net income
    'atq', # Total asset
    'teqq', # Shareholder's equity
    'epspiy', # EPS(Basic) incl. Extraordinary items
    'ceqq', # Common Equity
    'cshoq', # Common Shares Outstanding
    'dvpspq', # Dividends per share
    'actq', # Current assets
    'lctq', # Current liabilities
    'cheq', # Cash & Equivalent
    'rectq', # Recievalbles
    'cogsq', # Cost of  Goods Sold
    'invtq', # Inventories
    'apq',# Account payable
    'dlttq', # Long term debt
    'dlcq', # Debt in current liabilites
    'ltq' # Liabilities
]

# Omit items that will not be used
fund_data = fund[items]

fund_data = fund_data.rename(columns={
    'datadate':'date', # Date
    'oiadpq':'op_inc_q', # Quarterly operating income
    'revtq':'rev_q', # Quartely revenue
    'niq':'net_inc_q', # Quartely net income
    'atq':'tot_assets', # Assets
    'teqq':'sh_equity', # Shareholder's equity
    'epspiy':'eps_incl_ex', # EPS(Basic) incl. Extraordinary items
    'ceqq':'com_eq', # Common Equity
    'cshoq':'sh_outstanding', # Common Shares Outstanding
    'dvpspq':'div_per_sh', # Dividends per share
    'actq':'cur_assets', # Current assets
    'lctq':'cur_liabilities', # Current liabilities
    'cheq':'cash_eq', # Cash & Equivalent
    'rectq':'receivables', # Receivalbles
    'cogsq':'cogs_q', # Cost of  Goods Sold
    'invtq':'inventories', # Inventories
    'apq': 'payables',# Account payable
    'dlttq':'long_debt', # Long term debt
    'dlcq':'short_debt', # Debt in current liabilites
    'ltq':'tot_liabilities' # Liabilities
})

# Calculate financial ratios
date = pd.to_datetime(fund_data['date'], format='%Y%m%d')

tic = fund_data['tic'].to_frame('tic')

# Profitability ratios
# Operating Margin
OPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='OPM')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        OPM[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        OPM.iloc[i] = np.nan
    else:
        OPM.iloc[i] = np.sum(fund_data['op_inc_q'].iloc[i - 3:i]) / np.sum(fund_data['rev_q'].iloc[i - 3:i])

# Net Profit Margin
NPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='NPM')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        NPM[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        NPM.iloc[i] = np.nan
    else:
        NPM.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i - 3:i]) / np.sum(fund_data['rev_q'].iloc[i - 3:i])

# Return On Assets
ROA = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='ROA')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        ROA[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        ROA.iloc[i] = np.nan
    else:
        ROA.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i - 3:i]) / fund_data['tot_assets'].iloc[i]

# Return on Equity
ROE = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='ROE')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        ROE[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        ROE.iloc[i] = np.nan
    else:
        ROE.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i - 3:i]) / fund_data['sh_equity'].iloc[i]

    # For calculating valuation ratios in the next subpart, calculate per share items in advance
# Earnings Per Share
EPS = fund_data['eps_incl_ex'].to_frame('EPS')

# Book Per Share
BPS = (fund_data['com_eq'] / fund_data['sh_outstanding']).to_frame('BPS')  # Need to check units

# Dividend Per Share
DPS = fund_data['div_per_sh'].to_frame('DPS')

# Liquidity ratios
# Current ratio
cur_ratio = (fund_data['cur_assets'] / fund_data['cur_liabilities']).to_frame('cur_ratio')

# Quick ratio
quick_ratio = ((fund_data['cash_eq'] + fund_data['receivables']) / fund_data['cur_liabilities']).to_frame('quick_ratio')

# Cash ratio
cash_ratio = (fund_data['cash_eq'] / fund_data['cur_liabilities']).to_frame('cash_ratio')

# Efficiency ratios
# Inventory turnover ratio
inv_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='inv_turnover')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        inv_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        inv_turnover.iloc[i] = np.nan
    else:
        inv_turnover.iloc[i] = np.sum(fund_data['cogs_q'].iloc[i - 3:i]) / fund_data['inventories'].iloc[i]

# Receivables turnover ratio
acc_rec_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='acc_rec_turnover')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        acc_rec_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        acc_rec_turnover.iloc[i] = np.nan
    else:
        acc_rec_turnover.iloc[i] = np.sum(fund_data['rev_q'].iloc[i - 3:i]) / fund_data['receivables'].iloc[i]

# Payable turnover ratio
acc_pay_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='acc_pay_turnover')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        acc_pay_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        acc_pay_turnover.iloc[i] = np.nan
    else:
        acc_pay_turnover.iloc[i] = np.sum(fund_data['cogs_q'].iloc[i - 3:i]) / fund_data['payables'].iloc[i]

## Leverage financial ratios
# Debt ratio
debt_ratio = (fund_data['tot_liabilities'] / fund_data['tot_assets']).to_frame('debt_ratio')

# Debt to Equity ratio
debt_to_equity = (fund_data['tot_liabilities'] / fund_data['sh_equity']).to_frame('debt_to_equity')


# Create a dataframe that merges all the ratios
ratios = pd.concat([date,tic,OPM,NPM,ROA,ROE,EPS,BPS,DPS,
                    cur_ratio,quick_ratio,cash_ratio,inv_turnover,acc_rec_turnover,acc_pay_turnover,
                   debt_ratio,debt_to_equity], axis=1)

# Replace NAs infinite values with zero
final_ratios = ratios.copy()
final_ratios = final_ratios.fillna(0)
final_ratios = final_ratios.replace(np.inf,0)

list_ticker = df["tic"].unique().tolist()
list_date = list(pd.date_range(df['date'].min(),df['date'].max()))
combination = list(itertools.product(list_date,list_ticker))

# Merge stock price data and ratios into one dataframe
processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
processed_full = processed_full.merge(final_ratios,how='left',on=['date','tic'])
processed_full = processed_full.sort_values(['tic','date'])

# Backfill the ratio data to make them daily
processed_full = processed_full.bfill(axis='rows')
# Calculate P/E, P/B and dividend yield using daily closing price
processed_full['PE'] = processed_full['close']/processed_full['EPS']
processed_full['PB'] = processed_full['close']/processed_full['BPS']
processed_full['Div_yield'] = processed_full['DPS']/processed_full['close']
# Drop per share items used for the above calculation
processed_full = processed_full.drop(columns=['day','EPS','BPS','DPS'])

train = data_split(processed_full, '2009-01-01','2019-01-01')
trade = data_split(processed_full, '2019-01-01','2021-01-01')
# Check the length of the two datasets
print(len(train))
print(len(trade))


ratio_list = ['macd', 'rsi','cci', 'adx', 'cur_ratio', 'quick_ratio', 'cash_ratio', 'inv_turnover','acc_rec_turnover', 'acc_pay_turnover', 'debt_ratio', 'debt_to_equity',
       'PE', 'PB', 'Div_yield']

stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(ratio_list)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# Parameters for the environment
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": ratio_list,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4

}

# Establish the training environment using StockTradingEnv() class
e_train_gym = StockTradingEnv(df=train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

agent = DRLAgent(env = env_train)

model_a2c = agent.get_model("a2c")

trained_a2c = agent.train_model(model=model_a2c,tb_log_name='a2c',total_timesteps=100000)

agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
agent = DRLAgent(env = env_train)
model_ddpg = agent.get_model("ddpg")
trained_ddpg = agent.train_model(model=model_ddpg,tb_log_name='ddpg',total_timesteps=50000)



agent = DRLAgent(env = env_train)
TD3_PARAMS = {"batch_size": 100,
              "buffer_size": 1000000,
              "learning_rate": 0.001}
model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)
trained_td3 = agent.train_model(model=model_td3,tb_log_name='td3',total_timesteps=30000)


agent = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 1000000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)
trained_sac = agent.train_model(model=model_sac,tb_log_name='sac',total_timesteps=80000)



trade = data_split(processed_full, '2019-01-01','2021-01-01')
e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ddpg,
    environment = e_trade_gym)