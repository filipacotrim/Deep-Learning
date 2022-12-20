echo "learning rate evaluation"
python hw1-q2.py mlp -batch_size 16 -learning_rate 0.001 > results_2.2/rate=0.001.txt
echo "step"
python hw1-q2.py mlp -batch_size 16 -learning_rate 0.01 > results_2.2/rate=0.01.txt
echo "step"
python hw1-q2.py mlp -batch_size 16 -learning_rate 0.1 > results_2.2/rate=0.1.txt

echo "hidden size evalutation"
python hw1-q2.py mlp -batch_size 16 -hidden_sizes 100 > results_2.2/hidden=100.txt
echo "step"
python hw1-q2.py mlp -batch_size 16 -hidden_sizes 200 > results_2.2/hidden=200.txt

echo "dropout probability evaluation"
python hw1-q2.py mlp -batch_size 16 -dropout 0.3 > results_2.2/dropout=0.3.txt
echo "step"
python hw1-q2.py mlp -batch_size 16 -dropout 0.5 > results_2.2/dropout=0.5.txt

echo "activation function evaluation"
python hw1-q2.py mlp -batch_size 16 -activation relu > results_2.2/activation=relu.txt
echo "step"
python hw1-q2.py mlp -batch_size 16 -activation tanh > results_2.2/activation=tanh.txt

echo "number of layers"
python hw1-q2.py mlp -batch_size 16 -layers 2 > results_2.3/layers=2.txt
echo "step"
python hw1-q2.py mlp -batch_size 16 -layers 3 > results_2.3/layers=3.txt

echo "logistic regression"
python hw1-q2.py logistic_regression -learning_rate 0.001 > results_2.1/rate=0.001.txt
echo "step"
python hw1-q2.py logistic_regression -learning_rate 0.01 > results_2.1/rate=0.01.txt
echo "step"
python hw1-q2.py logistic_regression -learning_rate 0.1 > results_2.1/rate=0.1.txt