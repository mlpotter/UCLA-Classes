To run code:
	1. Open Anaconda Terminal
	2. Enter 'python hw4.py --num_epochs=25 --alpha=0.005 --verbose --save_fig --hidden=128'

More examples of running hw1.py:
	'python hw4.py --num_epochs=50 --save_fig'
	'python hw4.py --save_fig --verbose'
	'python hw4.py --alpha=0.0005'
	'python hw4.py --alpha=0.005 --save_fig --verbose --num_epochs=75'

Argument Comments:
--num_epochs: Number of Training Epochs, default=25)
--alpha: Learning Rate, default=0.005)
--verbose: True to print model training progress statistics (training accuracy/error, validation accuracy/error) each epoch, otherwise False, default=False
--save_fig: True to save the training curve figures (accuracy and loss curves), otherwise False,default=False
--hidden: Number of hidden units for the LSTM, default=128

If training curve figures are saved, they are saved in working directory of 'hw4.py' as jpg files: 
	 "Accuracy_History.jpg"
         "Cross_Entropy_History.jpg"

The best model based on the best cross-validation error during training (early stopping) is saved in the working directory of 'hw4.py':
	"best_model.pt"