--------------------------------------------------------------------------------
-- xtorch example training cifar10 whiten data.
-- data url: https://yadi.sk/d/em4b0FMgrnqxy
--------------------------------------------------------------------------------

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
dofile('./datagen/dataloader/plaindataloader.lua')

data = torch.load('./cifar10_whitened.t7')

trainloader = PlainDataLoader({ X=data.trainData.data, Y=data.trainData.labels })
testloader = PlainDataLoader({ X=data.testData.data, Y=data.testData.labels })

traindata = DataGen({
    dataloader=trainloader,
    randomflip=true,
    randomcrop=true
})

testdata = DataGen({ dataloader=testloader })

------------------------------------------------
-- 2. define net
--
dofile('augment.lua')
dofile('./models/resnet.lua')
dofile('./models/vgg.lua')
dofile('./models/googlenet.lua')

--net = getResNet()
net = getVGG()
--net = getGooglenet()

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
    nhorse = 1,   -- nb of threads to load data
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
