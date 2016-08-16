# xtorch: Torch extension for easy model training & test

## xtorch-dataset
xtorch dataset module, including `plaindataset`, `listdataset`, `classdataset`, which provides a unified interface for training.  

### `plaindataset`
Wraps `X` and `Y`. Suitable for small dataset that can be loaded in memory at once.  

```lua
dofile('plaindataset.lua')
ds = PlainDataset({
    X = samples,
    Y = targets
})
```

### `listdataset`
Loads images from disk dynamically with a multithread data loader. Suitable for big dataset that cannot fit in memory.  
```lua
dofile('listdataset.lua')
ds = ListDataset({
    directory = '/search/ssd/liukuang/cifar10/train/',
    list = '/search/ssd/liukuang/cifar10/train.txt',
    imsize = 32
})
```

- `directory` is a folder containing images.  
- `list` is a list files containing the image names and labels/targets separated by spaces.  

### `classdataset`
Loads training & test data from disk. But unlike `listdataset`, there is no index list,
the images are organized in subfolders, the subfolder names are the class names.
```lua
dofile('classdataset.lua')
ds = ClassDataset({
    directory = '/search/ssd/liukuang/cifar10/train/',
    imsize = 32
})
```

**Directory Structure**  
```
+-- train  
|  +-- class 1
|  |  +-- a.jpg
|  |  ...
|  +-- class 2
|  |  +-- b.jpg
|  |  ...
|  ...
+-- test  
|  +-- class 1
|  |  +-- c.jpg
|  |  ...
|  +-- class 2
|  |  +-- d.jpg
|  |  ...
|  ...
|
```

## xtorch.fit
`xtorch.fit(opt)` with `opt`:
- model options:
    - `net`: network to fit
- data options:  
    - `traindata`: train dataset  
    - `testdata`: test dataset  
    - `nhorse`: nb of threads to load data  
- training options:
    - `batchSize`: batch size
    - `nEpoch`: nb of epochs to train
    - `nClass`: nb of classes
- optimization options:
    - `optimizer`: optimization algorithm
    - `optimState`: optimization params
    - `criterion`: criterion defined
- general options:
    - `backend`: use CPU/GPU
    - `nGPU`: nb of GPUs to use  
    - `resume`: resume from checkpoint  
    - `verbose`: show debug info

Example:  
```lua
optimState = {
    learningRate = 0.001,
    learningRateDecay = 1e-7,
    weightDecay = 1e-4,
    momentum = 0.9,
    nesterov = true,
    dampening = 0.0
}

opt = {
    ----------- net options --------------------
    net = net,
    ----------- data options -------------------
    traindata = traindata,
    testdata = testdata,
    nhorse = 8,   -- nb of threads to load data, default 1
    ----------- training options ---------------
    batchSize = 128,
    nEpoch = 500,
    nClass = 10,
    ----------- optimization options -----------
    optimizer = optim.adam,
    criterion = nn.CrossEntropyCriterion,
    optimState = optimState,
    ----------- general options ----------------
    backend = 'GPU',    -- CPU or GPU, default CPU
    nGPU = 4,
    resume = true,
    verbose = true
}

xtorch.fit(opt)
```

## log & checkpoint  
- `utils.log`: automatically adding log while training.  
- `utils.saveCheckpoint`: saves checkpoint to disk.  
- `utils.loadCheckpoint`: loads saved checkpoint when `opt.resume=true`.  
