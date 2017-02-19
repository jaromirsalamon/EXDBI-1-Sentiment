import pandas as pd

train_o = pd.read_csv('data/in/en_sentiment.tsv', header=0, delimiter='\t', quoting=3)
train = train_o.loc[(train_o['sentiment'] != 'neutral') & (train_o['sentiment'] != 'na'), ['sentiment','tweet']]
print(train.groupby(['sentiment']).count())

test_1 = pd.read_csv('data/in/experiment-1_twitter.csv', header=0, delimiter=',', quoting=0)
test_2 = pd.read_csv('data/in/experiment-2_twitter.csv', header=0, delimiter=',', quoting=0)
frames = [test_1, test_2]
test = pd.concat(frames)

print(test.groupby(['sent_num']).count())