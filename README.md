# iconn - Interpretability in COnvolutional Neural Networks

This is an exploration of interpretability in Convolutional Neural Networks

## Running the code

### Checkout
```
$ git clone https://github.com/hnaik/iconn.git
$ cd iconn
```

### Install dependencies

```
$ pipenv install --python 3.7
$ pipenv shell
```

### Run code
```
$ PYTHONPATH=. apps/2d-shapes.py \
    --arch interpretable \
    --device cuda \
    --input-dir </path/to/data-dir> \
    --output-dir </path/to/output-dir> \
    --epochs 1 \
    --template-norm [original|l1|l2] \
    --template-cache-dir </path/to/cache-dir>
```
