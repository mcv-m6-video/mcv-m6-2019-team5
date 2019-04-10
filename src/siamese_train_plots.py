import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()


def main():
    nearest_neighbor_same_class_acc_s03_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S03_out_epochs_100_1554855902.809297-tag-nearest_neighbor_same_class_acc.csv'
    )
    nearest_neighbor_same_class_acc_s03_out = nearest_neighbor_same_class_acc_s03_out.assign(Name='S03_out')
    negative_classification_acc_s03_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S03_out_epochs_100_1554855902.809297-tag-negative_classification_acc.csv'
    )
    negative_classification_acc_s03_out = negative_classification_acc_s03_out.assign(Name='S03_out')
    train_loss_s03_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S03_out_epochs_100_1554855902.809297-tag-train_loss.csv'
    )
    train_loss_s03_out = train_loss_s03_out.assign(Name='S03_out')

    nearest_neighbor_same_class_acc_s01_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S01_out_epochs_100_1554854160.5032706-tag-nearest_neighbor_same_class_acc.csv'
    )
    nearest_neighbor_same_class_acc_s01_out = nearest_neighbor_same_class_acc_s01_out.assign(Name='S01_out')
    negative_classification_acc_s01_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S01_out_epochs_100_1554854160.5032706-tag-negative_classification_acc.csv'
    )
    negative_classification_acc_s01_out = negative_classification_acc_s01_out.assign(Name='S01_out')
    train_loss_s01_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S01_out_epochs_100_1554854160.5032706-tag-train_loss.csv'
    )
    train_loss_s01_out = train_loss_s01_out.assign(Name='S01_out')

    nearest_neighbor_same_class_acc_s04_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S04_out_epochs_100_1554855902.8494158-tag-nearest_neighbor_same_class_acc.csv'
    )
    nearest_neighbor_same_class_acc_s04_out = nearest_neighbor_same_class_acc_s04_out.assign(Name='S04_out')
    negative_classification_acc_s04_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S04_out_epochs_100_1554855902.8494158-tag-negative_classification_acc.csv'
    )
    negative_classification_acc_s04_out = negative_classification_acc_s04_out.assign(Name='S04_out')
    train_loss_s04_out = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_S04_out_epochs_100_1554855902.8494158-tag-train_loss.csv'
    )
    train_loss_s04_out = train_loss_s04_out.assign(Name='S04_out')
    train_loss_all = pd.read_csv(
        '../datasets/siamese_train_data/run_siamese_w6_all_epochs_300_1554851062.653252-tag-train_loss.csv'
    )
    train_loss_all = train_loss_all.assign(Name='All')

    train_loss = pd.concat((train_loss_s01_out, train_loss_s03_out, train_loss_s04_out, train_loss_all))
    train_loss.rename(columns={'Value': 'Train loss'}, inplace=True)

    nearest_neighbor_same_class_acc = pd.concat((nearest_neighbor_same_class_acc_s01_out,
                                                 nearest_neighbor_same_class_acc_s03_out,
                                                 nearest_neighbor_same_class_acc_s04_out))

    negative_classification_acc = pd.concat((negative_classification_acc_s01_out,
                                             negative_classification_acc_s03_out,
                                             negative_classification_acc_s04_out))

    sns.lineplot(x="Step", y="Train loss", hue='Name',
                 data=train_loss[train_loss.Step <= 25])
    plt.show()

    sns.lineplot(x="Step", y="Value", hue='Name',
                 data=nearest_neighbor_same_class_acc[nearest_neighbor_same_class_acc.Step < 50])
    plt.show()

    sns.lineplot(x="Step", y="Value", hue='Name',
                 data=negative_classification_acc[negative_classification_acc.Step < 50])
    plt.show()


if __name__ == '__main__':
    main()
