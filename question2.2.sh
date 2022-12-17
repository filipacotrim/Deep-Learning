echo "learning rate evaluation"
python hw1-q2.py mlp -learning_rate 0.001
python hw1-q2.py mlp -learning_rate 0.01
python hw1-q2.py mlp -learning_rate 0.1

echo "hidden size evalutation"
python hw1-q2.py mlp -hidden_sizes 100
python hw1-q2.py mlp -hidden_sizes 200

echo "dropout probability evaluation"
python hw1-q2.py mlp -dropout 0.3
python hw1-q2.py mlp -dropout 0.5

echo "activation function evaluation"
python hw1-q2.py mlp -activation relu
python hw1-q2.py mlp -activation tanh

