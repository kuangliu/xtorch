--------------------------------------------------------------------------------
-- This is a modified version of VGG network in
-- https://github.com/szagoruyko/cifar.torch
-- Modifications:
--  - removed dropout
--  - replace linear layers with convolutional layers and avg-pooling
--------------------------------------------------------------------------------
require 'nn'

function getVGG()
    local net = nn.Sequential()

    -- building block
    local function Block(nInputPlane, nOutputPlane)
        net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
        net:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
        net:add(nn.ReLU(true))
        return net
    end

    local function MP()
        net:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
        return net
    end

    local function Group(ni, no, N, f)
        for i=1,N do
            Block(i == 1 and ni or no, no)
        end
        if f then f() end
    end

    Group(3,64,2,MP)
    Group(64,128,2,MP)
    Group(128,256,4,MP)
    Group(256,512,4,MP)
    Group(512,512,4)
    net:add(nn.SpatialAveragePooling(2,2,2,2):ceil())
    net:add(nn.View(-1):setNumInputDims(3))
    net:add(nn.Linear(512, 10))

    return net
end
