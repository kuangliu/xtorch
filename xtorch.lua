local xtorch = {}
local opt

function xtorch.fit(opt_)
    opt = opt_
    if opt.backend == 'GPU' then
        require 'cunn'
        require 'cutorch'
        cudnn = require 'cudnn'
    end

    -- init model
    if opt.resume then  -- load checkpoint
        xtorch.resume()
    else                -- load new model
        xtorch.init()
    end

    parameters, gradParameters = net:getParameters()
    confusion = optim.ConfusionMatrix(opt.nClass)
    criterion = opt.backend=='GPU' and opt.criterion():cuda() or opt.criterion()

    -- init data loader/horses
    xtorch.initDataLoader()

    -- main loop
    for i = 1, opt.nEpoch do
        xtorch.train()
        xtorch.test()
    end
end

----------------------------------------------------------------
-- resume model from checkpoint
--
function xtorch.resume()
    print('==> resuming from checkpoint..')
    local latest = utils.loadCheckpoint()
    epoch = latest.epoch
    net = torch.load(latest.modelfile)
    optimState = torch.load(latest.optimfile)
    bestAcc = latest.bestAcc
    print(net)
end

----------------------------------------------------------------
-- init new model
--
function xtorch.init()
    print('==> init new model..')
    net = opt.net
    -- use GPU
    if opt.backend == 'GPU' then
        cudnn.convert(net, cudnn):cuda()
        cudnn.fastest = true
        cudnn.benchmark = true

        if opt.nGPU == 1 then
            cutorch.setDevice(1)
        else
            net = utils.makeDataParallelTable(net, opt.nGPU)
        end

        -- insert data casting layer
        local net_ = nn.Sequential()
                    :add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
                    :add(net)
        net = net_
    end

    net = utils.MSRinit(net)
    print(net)
end

----------------------------------------------------------------
-- init data loader
--
function xtorch.initDataLoader()
    if opt.nhorse == 1 or opt.nhorse == nil then   -- default single thread
        horses = {}
        function horses:addjob(f1, f2) f2(f1()) end
        function horses:synchronize() end
    else                                           -- multi thread
        threads = require 'threads'
        horses = threads.Threads(opt.nhorse,
            function()
                require 'nn'
                require 'torch'
            end,
            function(idx)
                print('init thread '..idx)
                dofile('datagen/datagen.lua')
                dofile('datagen/dataloader/classdataloader.lua')
                dofile('datagen/dataloader/listdataloader.lua')
                dofile('datagen/dataloader/plaindataloader.lua')
            end
        )
    end
end

----------------------------------------------------------------
-- cuda synchronize for each epoch/batch training/test.
--
function xtorch.cudaSync()
    if opt.backend=='GPU' then cutorch.synchronize() end
end

----------------------------------------------------------------
-- training
--
function xtorch.train()
    -- cuda sync for each epoch
    xtorch.cudaSync()
    net:training()

    -- parse arguments
    local nEpoch = opt.nEpoch
    local batchSize = opt.batchSize
    local optimState = opt.optimState
    local dataset = opt.traindata
    local c = require 'trepl.colorize'

    -- epoch tracker
    epoch = (epoch or 0) + 1
    print(string.format(c.Cyan 'Epoch %d/%d', epoch, nEpoch))

    -- do one epoch
    trainLoss = 0
    local epochSize = math.floor(dataset.N/opt.batchSize)
    local bs = opt.batchSize
    for i = 1,epochSize do
        horses:addjob(
            -- the job callback (runs in data-worker thread)
            function()
                local inputs, targets = dataset:sample(bs)
                return inputs, targets
            end,
            -- the end callback (runs in the main thread)
            function (inputs, targets)
                -- cuda sync for each batch
                xtorch.cudaSync()
                -- if use GPU, convert to cuda tensor
                targets = opt.backend=='GPU' and targets:cuda() or targets

                feval = function(x)
                    if x~= parameters then
                        parameters:copy(x)
                    end

                    net:zeroGradParameters()
                    local outputs = net:forward(inputs)
                    local f = criterion:forward(outputs, targets)
                    local df_do = criterion:backward(outputs, targets)
                    net:backward(inputs, df_do)
                    trainLoss = trainLoss + f

                    -- display progress & loss
                    confusion:batchAdd(outputs, targets)
                    confusion:updateValids()
                    utils.progress(i, epochSize, trainLoss/i, confusion.totalValid)
                    return f, gradParameters
                end
                opt.optimizer(feval, parameters, optimState)
                xtorch.cudaSync()
            end
        )
    end

    -- sync
    horses:synchronize() -- wait all horses back
    xtorch.cudaSync()

    -- cache for logging
    trainLoss = trainLoss/epochSize
    trainAcc = confusion.totalValid

    -- reset confusion for test
    if opt.verbose then print(confusion) end
    confusion:zero()
end

----------------------------------------------------------------
-- test
--
function xtorch.test()
    xtorch.cudaSync()
    net:evaluate()

    local dataset = opt.testdata
    local epochSize = math.floor(dataset.N/opt.batchSize)
    local bs = opt.batchSize

    testLoss = 0
    for i = 1,epochSize do
        horses:addjob(
            function()
                local inputs, targets = dataset:get(bs*(i-1)+1, bs*i)
                return inputs, targets
            end,
            function(inputs, targets)
                xtorch.cudaSync()
                targets = opt.backend=='GPU' and targets:cuda() or targets

                local outputs = net:forward(inputs)
                local f = criterion:forward(outputs, targets)
                testLoss = testLoss + f

                -- display progress
                confusion:batchAdd(outputs, targets)
                confusion:updateValids()
                utils.progress(i, epochSize, testLoss/i, confusion.totalValid)
                xtorch.cudaSync()
            end
        )
    end

    -- sync
    horses:synchronize()
    xtorch.cudaSync()

    -- logging
    testLoss = testLoss/epochSize
    testAcc = confusion.totalValid
    utils.log{trainLoss, testLoss, 100*trainAcc, 100*testAcc}

    -- save checkpoint
    bestAcc = bestAcc or -math.huge
    if confusion.totalValid > bestAcc then
        print('saving..')
        bestAcc = confusion.totalValid
        utils.saveCheckpoint(net, epoch, optimState, bestAcc)
    end

    -- reset for next epoch
    if opt.verbose then print(confusion) end
        confusion:zero()
        print('\n')
    end

return xtorch
