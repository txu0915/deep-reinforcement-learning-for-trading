from utils import *
from collections import Counter
from preprocessing.pull_and_process import *

data = preprocess_data(tic_list)
data = add_turbulence(data)
data.to_csv(f"data/done_data{now}.csv")
#data = pd.read_csv(f"data/done_data{20220201}.csv")

all_dates = data.datadate.unique()
num_dates = sorted([[k,v] for k, v in Counter(data.groupby('tic').count().sort_values('datadate').datadate).items()],
                   key=lambda x:x[1], reverse=True)[0][0]
tics_data_quality = data.groupby('tic').count().sort_values('datadate').datadate==num_dates
tics_complete_data = tics_data_quality.index[tics_data_quality==True]
data['filtered_stocks'] = data['tic'].apply(lambda x: x in tics_complete_data and x in config.tic_list)
df = data.loc[data.loc[:,'filtered_stocks']==True]
df = df.rename(columns={'datadate':'date', 'adjcp':'close'})

train = data_split(df, 20090101,20210101)
trade = data_split(df, 20210101,20220201)

ratio_list = ['macd', 'rsi','cci', 'adx']
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

model_catalog = {}
# ## train a2c
# agent = DRLAgent(env = env_train)
# model_a2c = agent.get_model("a2c")
# trained_a2c = agent.train_model(model=model_a2c,tb_log_name='a2c',total_timesteps=100000)
# model_catalog['a2c'] = trained_a2c

## train ppo
agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
trained_ppo = agent.train_model(model=model_ppo,tb_log_name='ppo',total_timesteps=50000)
model_catalog['ppo'] = trained_ppo

# ## train ddpg
# agent = DRLAgent(env = env_train)
# model_ddpg = agent.get_model("ddpg")
# trained_ddpg = agent.train_model(model=model_ddpg,tb_log_name='ddpg',total_timesteps=50000)
# model_catalog['ddgp'] = trained_ddpg
#
# ## train td3
# agent = DRLAgent(env = env_train)
# TD3_PARAMS = {"batch_size": 100,
#               "buffer_size": 1000000,
#               "learning_rate": 0.001}
# model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)
# trained_td3 = agent.train_model(model=model_td3,tb_log_name='td3',total_timesteps=30000)
# model_catalog['td3'] = trained_td3
#
# ## train sac
# agent = DRLAgent(env = env_train)
# SAC_PARAMS = {
#     "batch_size": 128,
#     "buffer_size": 1000000,
#     "learning_rate": 0.0001,
#     "learning_starts": 100,
#     "ent_coef": "auto_0.1",
# }
# model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)
# trained_sac = agent.train_model(model=model_sac,tb_log_name='sac',total_timesteps=80000)
# model_catalog['sac'] = trained_sac

## trade
now = datetime.datetime.now().strftime('%m%d%Y')
for selected_model, model in model_catalog.items():
    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model,environment = e_trade_gym)
    actions_last_day = pd.DataFrame({'tic': df_actions.columns,'score':df_actions.iloc[-1,:]}).sort_values(by='score')
    # actions_last_day.loc[:,'action_recommended'] = pd.qcut(actions_last_day.loc[:,'score'], q=[0, .33, 0.67, 1.],retbins=True,
    #                                                        labels = ['sell','hold','buy'],duplicates='drop')
    df_account_value.to_csv(f'results/final-df-account-vals-{selected_model}-{now}.csv',index=False)
    df_actions.to_csv(f'results/final-df-actions-{selected_model}-{now}.csv',index=False)
    actions_last_day.to_csv(f'results/final-actions-lastday-{selected_model}-{now}.csv',index=False)