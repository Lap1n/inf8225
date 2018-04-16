import csv
import numpy as np


class GoldDataLoader(object):
    def __init__(self, data_file_name="gc_1y_1min.csv", input_size=45, n_input_momentum=5, tick_size=5):
        # self.dones = []
        self.x = np.zeros(0)
        self.z = np.zeros(0)
        self.input_price_size = input_size
        self.input_momentum = n_input_momentum
        self.tick_size = tick_size

        self.index = 0
        self.current_end_day_index = 0
        self.current_total_reward = 0.0
        self.current_start_day_index = 0

        self.last_total_reward = 0.0
        self.current_total_number_of_trades = 0
        self.last_total_number_of_trades = 0
        self.MIN_BAR_PER_DAY = int(450 / self.tick_size)

        self.arranged_data = np.zeros(
            (250, int(self.MIN_BAR_PER_DAY - self.input_price_size), input_size + n_input_momentum))

        self.format_data(data_file_name)

        self.train_set, self.test_set = self.non_shuffling_train_test_split(self.arranged_data, test_size=0.3)

        # self.train_set = self.arranged_data

        # if shuffling_data_set is True:
        #     np.random.shuffle(self.train_set)
        #     np.random.shuffle(self.test_set)

        self.train_set = self.train_set
        self.test_set = self.test_set

    def non_shuffling_train_test_split(self,X, test_size=0.2):
        i = int((1 - test_size) * X.shape[0]) + 1
        X_train, X_test = np.split(X, [i])
        return X_train, X_test

    def norm(self, x, min_max):
        (min_, max_) = min_max
        return (x - min_) / (max_ - min_)

    def denorm(self, x):
        (min_, max_) = self.z_min_max
        return min_ + x * (max_ - min_)

    def format_data(self, data_file_name):
        with open("./data/" + data_file_name) as csvfile:
            reader = csv.DictReader(csvfile)
            lastTick = None
            counter = 0
            for row in reader:
                currentTick = float(row["close"])
                if lastTick is not None and counter % self.tick_size == 0:
                    tick_delta = currentTick - lastTick
                    self.x = np.append(self.x, currentTick)
                    self.z = np.append(self.z, tick_delta)
                counter += 1
                lastTick = currentTick
                pass
            pass
        pass
        """v = np.var(z)
        mean = np.mean(z)
    
        z = (z - mean)/ v"""
        min_ = np.min(self.z)
        max_ = np.max(self.z)

        self.z_min_max = (min_, max_)

        self.z = self.norm(self.z, self.z_min_max)
        self.arrange_data()

    def arrange_data(self):
        self.index = self.input_price_size
        self.current_end_day_index = self.MIN_BAR_PER_DAY
        for day_index in range(250):
            for i in range(0, int(self.MIN_BAR_PER_DAY - self.input_price_size)):
                next_z = self.z[int(self.index) - self.input_price_size: int(self.index)]
                next_x = self.x[int(self.index) - self.input_price_size: int(self.index)]

                current_price = next_x[-1]
                momentum_3h = current_price - self.x[
                    max(int(self.index - 180 / self.tick_size), self.current_start_day_index)]
                momentum_5h = current_price - self.x[
                    max(int(self.index - 300 / self.tick_size), self.current_start_day_index)]
                momentum_1d = current_price - self.x[max(int(self.index - 450 / self.tick_size), 0)]
                momentum_3d = current_price - self.x[max(int(self.index - 1350 / self.tick_size), 0)]
                momentum_10d = current_price - self.x[max(int(self.index - 4500 / self.tick_size), 0)]

                next_z = np.insert(next_z, 0, [momentum_3h, momentum_5h, momentum_1d, momentum_3d, momentum_10d])
                self.arranged_data[day_index, i] = next_z
                self.index += 1
            self.next_day()
            pass
        pass

    pass

    def next_day(self):
        self.index = self.current_end_day_index + self.input_price_size
        self.current_start_day_index = self.current_end_day_index
        self.current_end_day_index += self.MIN_BAR_PER_DAY

    def get_training_data(self):
        return self.train_set

    def get_test_data(self):
        return self.test_set
