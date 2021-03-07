To run code:
	1. Open Anaconda Terminal
	2. Enter 'python hw3.py --model_choice="CNN+FCNN" --num_epochs=100 --save_fig --verbose --activation'

More examples of running hw1.py:
	'python hw3.py --model_choice="CNN+FCNN" --num_epochs=100 --activation --save_fig --verbose'
	'python hw3.py --model_choice="FCNN" --num_epochs=75'
	'python hw3.py --model_choice="CNN+FCNN"'
	'python hw3.py --alpha=0.005 --save_fig --threshold=0.85'
	'python hw3.py --model_choice="CNN+FCNN" --num_epochs=100 --save_fig'

Argument Comments:
--model_choice: Model selection of FCNN (Fully Connected Neural Network) or CNN+FCNN (Convolutional Neural Network + CNN)', default='CNN+FCNN'
--num_epochs: Number of Training Epochs, default=100)
--activation': True to use ReLU activations after each hidden layer, otherwise False, default=False
--alpha: 'Learning Rate', default=.05)
--verbose: True to print model training progress statistics (training accuracy/error, validation accuracy/error), otherwise False, default=False
--save_fig: True to save the training curve figures (accuracy and loss curves), otherwise False,default=False
--threshold: Accuracy threshold for model to beat (will be plotted as line on accuracy learning curves), default=.85

If training curve figures are saved, they are saved in working directory of 'hw3.py' as jpg files: 
	 "[model_choice]_[activation]_acc_history.jpg"
         "[model_choice]_[activation]_error_history.jpg"

The best model based on the cross-validation error is saved in the working directory of 'hw3.py':
	"[model_choice"]_[activations].pt"