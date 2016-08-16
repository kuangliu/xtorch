require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'pl'
require 'paths'

utils = dofile('utils.lua')
xtorch = dofile('xtorch.lua')
------------------------------------------------
-- 1. prepare data
--
dofile('./xtorch-dataset/classdataset.lua')
traindata = ClassDataset({
   directory = '/search/ssd/liukuang/image/train',
   imsize = 32,
   transform = {
       standardize = true
   }
})

-- utilize the training mean & std to test dataset
mean,std = traindata:calcMeanStd()

testdata = ClassDataset({
   directory = '/search/ssd/liukuang/image/test',
   imsize = 32,
   mean = mean,
   std = std,
   transform = {
       standardize = true
   }
})

paths.mkdir('cache')
torch.save('./cache/traindata.t7',traindata)
torch.save('./cache/testdata.t7',testdata)
--traindata = torch.load('./cache/traindata.t7')
--testdata = torch.load('./cache/testdata.t7')

------------------------------------------------
-- 2. define net
--
dofile('augment.lua')
dofile('./models/resnet.lua')
dofile('./models/vgg.lua')
--net = getResNet()
net = getVGG()

------------------------------------------------
-- 3. init optimization params
--
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

------------------------------------------------
-- 4. and fit!
--
xtorch.fit(opt)
