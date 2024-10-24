import dill
import pandas as pd
import tzlocal
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime

sched = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv('data/homework.csv')
file_name = "cars_pipe.pkl"
with open(file_name, 'rb') as file:
    model = dill.load(file)


@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(frac=0.0005)
    data['preds'] = model['model'].predict(data)
    print(data[['id', 'preds', 'price']])
    print(f'{datetime.now()}: OK')


if __name__ == '__main__':
    sched.start()
