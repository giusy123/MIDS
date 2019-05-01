import matplotlib.pyplot as plt
import seaborn as sns
import os

class Plot():
    def __init__(self, pathPlot):
        self._pathPlot = pathPlot

    def plotStatistics(self, train, test, cls):
        s = train[cls]
        s = s.astype('category')
        s = s.cat.rename_categories(["Attacks", "Normal"])
        s2 = test[cls]
        s2 = s2.astype('category')
        s2 = s2.cat.rename_categories(["Attacks", "Normal"])

        fig = plt.figure()
        ax = plt.subplot(1, 2, 1)
        ax = sns.countplot(x=s)
        ax.set_title('Train set')
        ncount = len(s)

        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                        ha='center', va='bottom')

        ax2 = plt.subplot(1, 2, 2)
        ax2 = sns.countplot(x=s2)
        ax2.set_title('Test set')
        ncount2 = len(s2)

        for p in ax2.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax2.annotate('{:.1f}%'.format(100. * y / ncount2), (x.mean(), y),
                         ha='center', va='bottom')
        plt.subplots_adjust(bottom=0.15, wspace=0.4)
        plt.savefig(os.path.join(self._pathPlot,'statistics.png'))
        plt.show()
        plt.close()


    def printPlotLoss(self,history, d):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(os.path.join(self._pathPlot,"plotLoss" + str(d) + ".png"))
        plt.close()

    def printPlotAccuracy(self,history, d):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(os.path.join(self._pathPlot,"plotAccuracy" + str(d) + ".png"))
        plt.close()

