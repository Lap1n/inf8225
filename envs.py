from projet.a2cGold.TradingEnv import TradingEnv


def make_env(data_set):
    def _thunk():
        env = TradingEnv(data_set)
        return env

    return _thunk
