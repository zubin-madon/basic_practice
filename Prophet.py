import pandas as pd
import numpy as np
import quandl as qd
pd.set_option("display.precision", 8)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
import fbprophet
from matplotlib import pyplot as plt
from scipy.stats import boxcox


class HxroProphet:

    def execute(self, product='BCHAIN/MKPRU'):
        KEY = "k_S8LgNpXGf54_UgGbL-"
        qd.ApiConfig.api_key = KEY
        if product == 'hashrate':
            product = 'BCHAIN/HRATE'
        elif product == 'volume':
            product = 'BCHAIN/TRVOU'
        elif product == 'difficulty':
            product = 'BCHAIN/DIFF'
        elif product == 'blocksize':
            product = 'BCHAIN/AVBLS'
        elif product == 'vix':
            product = 'CHRIS/CBOE_VX1'
        btc = qd.get(product)
        if product == 'CHRIS/CBOE_VX1':
            btc.reset_index(inplace=True)
            btc = btc[['Trade Date', 'Close']]
            btc.columns = (['Date', 'Value'])
            btc = btc.iloc[443:]
            # print(btc)
            # btc.set_index('Date', inplace=True)
            btc.set_index('Date', inplace=True)
            btc = btc[btc['Value'] > 0]
            btc['log_y'], lam = boxcox(btc['Value'])
            btc['Date'] = btc.index

        # print(btc)
        if product != 'CHRIS/CBOE_VX1':
            btc = btc.loc[(btc != 0).any(1)]
            btc = btc.iloc[:]
            btc['Date'] = btc.index
            btc['log_y'], lam = boxcox(btc['Value'])


        # btc["log_y"] = np.log(btc["Value"])
        btc = btc.rename(columns={"Date": "ds", "log_y" : "y"})
        # btc = btc.iloc[:-300]
        # print(btc)
        priors = [0.001, 0.0025]
        prophets, labels = [], []
        for prior in priors:
            prophet = fbprophet.Prophet(changepoint_prior_scale=prior)
            prophet.fit(btc)

            prophets.append(prophet)
            labels.append(r"CP Prior = " + str(prior))

        forecasts = []
        for prophet in prophets:
            if product != 'CHRIS/CBOE_VX1':
                forecast = prophet.make_future_dataframe(periods=365*2, freq="D")
                forecast = prophet.predict(forecast)

                forecast = forecast.rename(columns={"ds": str(priors[prophets.index(prophet)]) + "_ds"})
                forecasts.append(forecast)
            else:
                forecast = prophet.make_future_dataframe(periods=7 * 2, freq="D")
                forecast = prophet.predict(forecast)

                forecast = forecast.rename(columns={"ds": str(priors[prophets.index(prophet)]) + "_ds"})
                forecasts.append(forecast)
        # print(btc)
        from scipy.special import inv_boxcox
        output = pd.merge(forecasts[0], forecasts[1], how = "inner", left_on = "0.001_ds", right_on = "0.0025_ds")
        output = output.rename(columns={"0.001_ds": "Date"}).drop("0.0025_ds", axis=1)
        output = output.set_index('Date')
        output[['yhat_x','yhat_upper_x','yhat_lower_x', 'yhat_y','yhat_upper_y','yhat_lower_y']] = output[['yhat_x','yhat_upper_x','yhat_lower_x', 'yhat_y','yhat_upper_y','yhat_lower_y']].apply(lambda x: inv_boxcox(x, lam))


        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_facecolor('black')
        fig.set_facecolor('black')
        # fig = plt.figure(facecolor='black', figsize=(12, 8))
        # ax = fig.add_axes()
        ax.plot(output.index, output["yhat_x"], label=labels[0], color='#fc0a9a')
        ax.fill_between(output.index, output["yhat_upper_x"], output["yhat_lower_x"], alpha=0.25, edgecolor = "#fc0a9a", facecolor='.75')
        ax.plot(output.index, output["yhat_y"], "r", label=labels[1], color='#00ff00')
        ax.fill_between(output.index, output["yhat_upper_y"], output["yhat_lower_y"], alpha=0.25, edgecolor = "#00ff00", facecolor='.75')
        # ax.plot(btc.ds, inv_boxcox(btc.y, lam), color="white", linewidth=3, label=r"Bitcoin Price")
        # a=ax.get_yticks().tolist()
        # ax.set_yticklabels(np.round(np.exp(a), 1))
        # ax.set_ylim(1000, 50000)
        plt.legend(loc="upper left")
        if product == 'BCHAIN/MKPRU':
            ax.plot(btc.ds, inv_boxcox(btc.y, lam), color="white", linewidth=3, label=r"Bitcoin Price")
            a = ax.get_yticks().tolist()
            ax.set_title('BTC/USD Prophet Model', color='0.75', fontsize=18)
            plt.ylabel("BTC PRICE IN USD", fontsize=12, color='0.75')
            plt.yscale('log')
        elif product == 'BCHAIN/HRATE':
            ax.plot(btc.ds, inv_boxcox(btc.y, lam), color="white", linewidth=3, label=r"Bitcoin Hashrate")
            a = ax.get_yticks().tolist()
            ax.set_title('BTC Hashrate Prophet Model', color='0.75', fontsize=18)
            plt.ylabel("BTC Hashrate (Megahash/S)", fontsize=12, color='0.75')
            plt.yscale('log')
        elif product == 'BCHAIN/TRVOU':
            ax.plot(btc.ds, inv_boxcox(btc.y, lam), color="white", linewidth=3, label=r"Bitcoin Volume")
            a = ax.get_yticks().tolist()
            ax.set_title('BTC Volume Prophet Model', color='0.75', fontsize=18)
            plt.ylabel("BTC VOLUME IN USD", fontsize=12, color='0.75')
            plt.yscale('log')
        elif product == 'BCHAIN/DIFF':
            ax.plot(btc.ds, inv_boxcox(btc.y, lam), color="white", linewidth=3, label=r"Bitcoin Difficulty")
            a = ax.get_yticks().tolist()
            ax.set_title('BTC Difficulty Prophet Model', color='0.75', fontsize=18)
            plt.ylabel("BTC DIFFICULTY", fontsize=12, color='0.75')
            plt.yscale('log')
        elif product == 'BCHAIN/AVBLS':
            ax.plot(btc.ds, inv_boxcox(btc.y, lam), color="white", linewidth=3, label=r"Bitcoin Blocksize")
            a = ax.get_yticks().tolist()
            ax.set_title('BTC Avg Blocksize Prophet Model', color='0.75', fontsize=18)
            plt.ylabel("BTC BLOCKSIZE (MB)", fontsize=12, color='0.75')
            plt.yscale('linear')
        elif product == 'CHRIS/CBOE_VX1':
            ax.plot(btc.ds, inv_boxcox(btc.y, lam), color="white", linewidth=3, label=r"VX1")
            a = ax.get_yticks().tolist()
            ax.set_title('VIX Prophet Model', color='0.75', fontsize=18)
            plt.ylabel("VIX VALUE", fontsize=12, color='0.75')
            plt.yscale('linear')
        ax.tick_params(axis='y', colors='0.75')
        ax.tick_params(axis='x', colors='0.75')
        plt.grid(color='.75', linestyle='-', linewidth=.5, axis='y')
        # plt.yscale('log')
        # plt.savefig('BTCProphet.png', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.show()
    # m = fbprophet.Prophet()
    # m.fit(btc)
    # future = m.make_future_dataframe(periods=365)
    # forecast = m.predict(future)
    # from scipy.special import inv_boxcox
    # forecast[['yhat','yhat_upper','yhat_lower']] = forecast[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))
    #
    # m.plot(forecast)
    # plt.show()

test = HxroProphet()
test.execute()
