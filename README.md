# Connoisseur

![Battle of Grunwald](assets/intro-battle-of-grunwald.jpg)
“Battle of Grunwald”, Jan Matejko, 1878. From Public Domain.  
Available at: [wikiart.org/en/jan-matejko/battle-of-grunwald-1878](https://www.wikiart.org/en/jan-matejko/battle-of-grunwald-1878).

This repository holds the experiments on Machine Learning applied to the analysis of painting provenance.  
Check the [INSTALL.md](docs/INSTALL.md) file for instructions on how to
prepare your environment to run connoisseur.  
Results can be seen at [REPORTS.md](docs/REPORTS.md).

## Associated Publications

- David L.O., Pedrini H., Dias Z., Rocha A. (2021) Authentication of Vincent van Gogh’s Work. In: Computer Analysis of Images and Patterns (CAIP). Lecture Notes in Computer Science, vol 13053. Springer, Cham.
- L. David, H. Pedrini, Z. Dias and A. Rocha. "Connoisseur: Provenance Analysis in Paintings," IEEE Symposium Series on Computational Intelligence (SSCI), 2021.

## Running Experiments

After entering the virtual environment or initiating the docker container,
experiments can be found at the `/connoisseur/experiments` folder. An
execution example follows:

```shell
cd ./experiments/
python 1-extract-patches.py with batch_size=256 image_shape=[299,299,3] \
                                 dataset_name='VanGogh' \
                                 data_dir="./datasets/vangogh" \
                                 saving_directory="./datasets/vangogh/random_299/" \
                                 valid_size=.25
```

This experiment will download, extract and prepare van Gogh's dataset into the
`data_dir` directory. Finally, it will extract patches from all samples and
save them in `saving_directory`.

Each experiment is wrapped by [sacred](http://sacred.readthedocs.io/) package,
capable of monitoring an experiment and logging its progression to a database
or file. To do so, use the `m` or `F` parameter:

```shell
python 1-extract-patches.py -m 107.0.0.1:27017:experiments  # Requires MongoDB
python 1-extract-patches.py -F ./extract-patches/
```
