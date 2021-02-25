from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



def evaluation_report(y_true,y_pred,labels):
    print(classification_report(y_true, y_pred,digits=6))
    
    
    #cm = confusion_matrix(y_true, y_pred, labels=labels)

    #disp = ConfusionMatrixDisplay(confusion_matrix=cm,
     #                             display_labels=labels)

    #disp.plot()
    #plt.show()