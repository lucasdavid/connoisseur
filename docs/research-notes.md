# Research Notes

## Possible Approaches

**Legend:**

```py
  default_params = {
    n_epochs: 1000000,
    batch_size: 32,
    learning_rate: .001,
    dropout: .5,
    checkpoint_every: 100,
    log_every: 100,
    train_validation_test_split: (.9, .1)
  }

  grid = {
    dataset: { paintings91, wikiart-paintings }
    architecture: { VGG, AlexNET, PigeoNET }
    mode: { freshly-trained, transfer-learning, hot-initialized }
    classifier: { softmax, SVM }
    task: { discrimination, attribution }
    classes-used: { 10%, half, all }
  }
```

**Reports:**

* **000012**: loss 4.5, valid-loss 4.5, prediction score: 5%.
Iterations completed: 100.000. Obs.: same result observed for batch-size 10.
* **010012**: loss 4.5, valid-loss 4.5, prediction score: 5%.
Iterations completed: 43.900.
