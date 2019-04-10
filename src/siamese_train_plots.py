import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def main():
    nearest_neighbor_same_class_acc = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S03_out_epochs_100_1554855902.809297-tag-nearest_neighbor_same_class_acc.csv'
    )

    negative_classification_acc = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S03_out_epochs_100_1554855902.809297-tag-negative_classification_acc.csv'
    )

    train_loss_S03_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S03_out_epochs_100_1554855902.809297-tag-train_loss.csv'
    )
    train_loss_S03_out = train_loss_S03_out.assign(Name='S03_out')

    train_loss_S01_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S01_out_epochs_100_1554854160.5032706-tag-train_loss.csv'
    )
    train_loss_S01_out = train_loss_S01_out.assign(Name='S01_out')

    train_loss_S04_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S04_out_epochs_100_1554855902.8494158-tag-train_loss.csv'
    )
    train_loss_S04_out = train_loss_S04_out.assign(Name='S04_out')

    train_loss = pd.concat((train_loss_S01_out, train_loss_S03_out, train_loss_S04_out))
    train_loss.rename(columns={'Value': 'Train loss'}, inplace=True)

    sns.lineplot(x="Step", y="Train loss", hue='Name',
                 data=train_loss[train_loss.Step < 50])
    plt.show()


if __name__ == '__main__':
    main()
