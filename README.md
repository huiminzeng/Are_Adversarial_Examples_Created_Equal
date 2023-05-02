# Are Adversarial Examples Created Equal? A Learnable Weighted Minimax Risk for Robustness under Non-uniform Attacks

This is the official code for the [AAAI'21 paper](https://ojs.aaai.org/index.php/AAAI/article/view/17292) "Are Adversarial Examples Created Equal? A Learnable Weighted Minimax Risk for Robustness under Non-uniform Attacks".

Adversarial Training is proved to be an efficient method to defend against adversarial examples, being one of the few defenses that withstand strong attacks. However, traditional defense mechanisms assume a uniform attack over the examples according to the underlying data distribution, which is apparently unrealistic as the attacker could choose to focus on more vulnerable examples. We present a weighted minimax risk optimization that defends against non-uniform attacks, achieving robustness against adversarial examples under perturbed test data distributions. Our modified risk considers importance weights of different adversarial examples and focuses adaptively on harder examples that are wrongly classified or at higher risk of being classified incorrectly. The designed risk allows the training process to learn a strong defense through optimizing the importance weights. The experiments show that our model significantly improves state-of-the-art adversarial accuracy under non-uniform attacks without a significant drop under uniform attacks.

## Citing 

Please consider citing the following paper if you use our methods in your research:
```
@inproceedings{zeng2021adversarial,
  title={Are adversarial examples created equal? A learnable weighted minimax risk for robustness under non-uniform attacks},
  author={Zeng, Huimin and Zhu, Chen and Goldstein, Tom and Huang, Furong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={12},
  pages={10815--10823},
  year={2021}
}
```

# Running Our Code
## Adversarial Training (AT_base)
- train robust network on MNIST/CIFAR10
```
python train_at_MNIST.py
python train_at_cifar10.py
```
- evaluate the trained models under PGD attacks
```
python eval_cifar10.py
python eval_mnist.py
```

## TRADES Adversarial Training (TRADES_base)
- train robust network on MNIST/CIFAR10
```
python train_trades_MNIST.py
python train_trades_cifar10.py
```
- evaluate the trained models under PGD attacks
```
python eval_cifar10.py
python eval_mnist.py
```
