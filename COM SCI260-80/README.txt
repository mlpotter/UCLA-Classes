To run code:
	1. Open Anaconda Terminal
	2. Enter 'python hw1.py --model_choice=LR --num_epochs=10 --momentum=0 --save_fig=True'

More examples of running hw1.py:
	'python hw1.py --model_choice=LR --num_epochs=10 --momentum=0 --save_fig=False'
	'python hw1.py --model_choice=SVM --num_epochs=10 --momentum=0.9 --save_fig=True'
	'python hw1.py --model_choice=SVM'
	'python hw1.py --model_choice=SVM --momentum=0.9'

Argument Comments:
	model_choice: Model selection of "LR" (Logistic Regression) or "SVM" (Support Vector Machine), default='LR'
	num_epochs: Number of Training Epochs as integer, default=10
	momentum: Momentum parameter value for Stochastic Gradient Descent (SGD) [0,1), default=0.0
	save_fig: True to save the training curve figure, otherwise False, default=False

If training curve figures are saved, they are saved in working directory of 'hw1.py' as png files. 
The file is saved as "[model_choice]_SGD-[momentum].png"