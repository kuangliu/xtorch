# xtorch: Torch extension for easy model training & test

## dataloader
`dataloader` loads samples & targets from files, including `classdataloader`, `listdataloader` and `plaindataloader`.

### `plaindataloader`
Loads `X` and `Y` that can be loaded in memory at once.  

```lua
dataloader = PlainDataloader({
    X = samples,
    Y = targets
})
```

### `listdataloader`
Loads images from disk dynamically with a multithread data loader. Suitable for big dataset that cannot fit in memory.  
```lua
dataloader = ListDataLoader({
    directory = '/search/ssd/liukuang/cifar10/train/',
    list = '/search/ssd/liukuang/cifar10/train.txt',
    imsize = 32
})
```

- `directory` is a folder containing images.  
- `list` is a list files containing the image names and labels/targets separated by spaces.  
- `imsize` is the image target size.

### `classdataloader`
Unlike `listdataset`, there is no index list needed,
the images are organized in subfolders, and the subfolder names are the class names.
```lua
dataloader = ClassDataLoader({
    directory = '/search/ssd/liukuang/cifar10/train/',
    imsize = 32
})
```

**Directory**  
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

## datagen
`datagen` performs a series of image processing functions, including zero mean, std normalization ...
The parameters including:
- `dataloader`: loads images
- `standardize`: perform zero-mean and std normalization
- `randomflip`: randomly flip inputs
- `randomcrop`: randomly crop inputs
- `mean`: input mean
- `std`: input std

Example:
```lua
traindata = DataGen({
    dataloader=trainloader,
    standardize=true,
    randomflip=true,
    randomcrop=true
})
mean,std = traindata:getmeanstd()

testdata = DataGen({
    dataloader=testloader,
    standardize=true,
    mean=mean,
    std=std
})
```

## xtorch.fit
`xtorch.fit(opt)` with `opt`:
- model options:
    - `net`: network to fit
- data options:  
    - `traindata`: train datagen  
    - `testdata`: test datagen  
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
    nhorse = 8,   -- nb of threads to load data
    ----------- training options ---------------
    batchSize = 128,
    nEpoch = 500,
    nClass = 10,
    ----------- optimization options -----------
    optimizer = optim.adam,
    criterion = nn.CrossEntropyCriterion,
    optimState = optimState,
    ----------- general options ----------------
    backend = 'GPU',    -- CPU or GPU
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
