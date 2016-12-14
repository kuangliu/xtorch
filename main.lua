require 'pl'
require 'nn'
require 'nnx'
require 'xlua'
require 'torch'
require 'optim'
require 'image'
require 'paths'

utils = dofile('utils.lua')
xtorch = dofile('xtorch.lua')

------------------------------------------------
-- 1. prepare data
--
dofile('./datagen/datagen.lua')
dofile('./dataloader/classdataloader.lua')

trainloader = ClassDataLoader {
    directory='/search/ssd/liukuang/image/train',
    imsize=32
}

testloader = ClassDataLoader {
    directory='/search/ssd/liukuang/image/test',
    imsize=32
}

traindata = DataGen {
    dataloader=trainloader,
    standardize=true,
    randomflip=true,
    randomcrop={ size=32, pad=4 }
}
mean,std = traindata:getmeanstd()

testdata = DataGen {
    dataloader=testloader,
    standardize={ mean=mean, std=std }
}

paths.mkdir('cache')
torch.save('./cache/traindata.t7',traindata)
torch.save('./cache/testdata.t7',testdata)
-- traindata = torch.load('./cache/traindata.t7')
-- testdata = torch.load('./cache/testdata.t7')

------------------------------------------------
-- 2. define net
--
dofile('./models/vgg.lua')
dofile('./models/resnet.lua')
dofile('./models/googlenet.lua')

-- net = getVGG()
-- net = getResNet()
net = getGooglenet()

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
    resume = false,
    verbose = true
}
opt = xlua.envparams(opt)

------------------------------------------------
-- 4. and fit!
--
xtorch.fit(opt)
