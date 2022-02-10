from collections import Counter
from model.models import *
# data = preprocess_data(tic_list)
# data = add_turbulence(data)
# data.to_csv(f"data/done_data{now}.csv")
data = pd.read_csv(f"data/done_data{20220201}.csv")



start = time.time()
trade_start_date = 20211001
validation_end_date = 20210930
validation_start_date = 20210701
training_end_date = 20210630

insample_turbulence = data[(data.datadate <= training_end_date) & (data.datadate>=20090000)]
insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

today = 20220127
unique_trade_date = data[(data.datadate > trade_start_date) & (data.datadate <= today)].datadate.unique()
print(len(unique_trade_date))
df = data

historical_turbulence = df.loc[df.loc[:,'datadate'] < trade_start_date, :]
historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

if historical_turbulence_mean > insample_turbulence_threshold:
    # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
    # then we assume that the current market is volatile,
    # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
    # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
    turbulence_threshold = insample_turbulence_threshold
else:
    # if the mean of the historical data is less than the 90% quantile of insample turbulence data
    # then we tune up the turbulence_threshold, meaning we lower the risk
    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
# print("turbulence_threshold: ", turbulence_threshold)

############## Environment Setup starts ##############
## training env
i = '000'
all_dates = df.datadate.unique()
num_dates = sorted([[k,v] for k, v in Counter(df.groupby('tic').count().sort_values('datadate').datadate).items()],
                   key=lambda x:x[1], reverse=True)[0][0]
tics_data_quality = df.groupby('tic').count().sort_values('datadate').datadate==num_dates
tics_complete_data = tics_data_quality.index[tics_data_quality==True]
df['filtered_stocks'] = df['tic'].apply(lambda x: x in tics_complete_data and x in tic_list)
df = df.loc[df.loc[:,'filtered_stocks']==True]

# counter = 0
# last_valid_training = all_dates[0]
# for cur_date in all_dates:
#     if len(df.loc[df.loc[:,'datadate']==cur_date,'tic'].unique()) != len(tic_list):
#         print(cur_date, len(df.loc[df.loc[:,'datadate']==cur_date,'tic'].unique()))
#         counter += 1
#         last_valid_training = cur_date
# print("full stock data valid from: ", last_valid_training, counter,"dates missing")


train = data_split(df, start=20140101, end=training_end_date)
validation = data_split(df, start=validation_start_date, end=validation_end_date)
env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

env_val = DummyVecEnv([lambda: StockEnvValidation(
    validation,turbulence_threshold=turbulence_threshold, iteration=i)])
obs_val = env_val.reset()

model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), timesteps=1000)
DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
sharpe_a2c = get_validation_sharpe(i)

model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=1000)
DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
sharpe_ppo = get_validation_sharpe(i)

model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=1000)
DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
sharpe_ddpg = get_validation_sharpe(i)

print(f"sharpe values, a2c, ppo, ddpq:{sharpe_a2c} | {sharpe_ppo} | {sharpe_ddpg}")

# Model Selection based on sharpe ratio
if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
    model_ensemble = model_ppo
elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
    model_ensemble = model_a2c
else:
    model_ensemble = model_ddpg

last_state_ensemble = DRL_prediction(df=df, model=model_ddpg, name="ensemble",
                                     last_state=[], iter_num=i,
                                     turbulence_threshold=turbulence_threshold,
                                     trade_start_date = trade_start_date,
                                     trade_end_date = today)

# a = pd.read_csv('results/final-actions-as-of-day-20220125-test.csv')
# b = pd.read_csv('results/final-actions-as-of-day-20220125.csv')
#
# sorted(a.sort_values('action').head(20).tic)
# sorted(b.sort_values('action').head(20).tic)
