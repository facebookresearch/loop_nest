# Setting up for utils

First install conda

```
HOME=/mnt/ssd1/josepablocam
pushd ${HOME}
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# follow instructions, agree to license etc
popd
```

Then use conda to import the environment file provided as

```
conda create -f env.yml
```

You can then activate it as

```
conda activate loop-nest-env
```

And you should be able to use the scripts in this folder

