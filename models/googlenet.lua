--------------------------------------------------------------------------------
-- GoogLeNet with BN
--------------------------------------------------------------------------------

require 'nn'

ReLU = nn.ReLU
Conv = nn.SpatialConvolution
MaxP = nn.SpatialMaxPooling
AvgP = nn.SpatialAveragePooling
BN = nn.SpatialBatchNormalization

function inception(nInputPlane, n1, n31, n33, n51, n53, nPool)
    local cat = nn.Concat(2)

    -- 1x1 branch
    local b1 = nn.Sequential()
                :add(Conv(nInputPlane,n1,1,1))
                :add(BN(n1,1e-3))
                :add(ReLU(true))

    -- 1x1-3x3 branch
    local b2 = nn.Sequential()
                :add(Conv(nInputPlane,n31,1,1))
                :add(BN(n31,1e-3))
                :add(ReLU(true))
                :add(Conv(n31,n33,3,3,1,1,1,1))
                :add(BN(n33,1e-3))
                :add(ReLU(true))

    -- 1x1-5x5 branch
    local b3 = nn.Sequential()
                :add(Conv(nInputPlane,n51,1,1))
                :add(BN(n51,1e-3))
                :add(ReLU(true))
                :add(Conv(n51,n53,5,5,1,1,2,2))
                :add(BN(n53,1e-3))
                :add(ReLU(true))

    -- 3x3pool-1x1 branch
    local b4 = nn.Sequential()
                :add(MaxP(3,3,1,1,1,1):ceil())
                :add(Conv(nInputPlane,nPool,1,1))
                :add(BN(nPool,1e-3))
                :add(ReLU(true))

    cat:add(b1):add(b2):add(b3):add(b4)
    return cat
end

function getGooglenet()
    local net = nn.Sequential()
    net:add(Conv(3,192,3,3,1,1,1,1))
        :add(BN(192,1e-3))
        :add(ReLU(true))

    local a3 = inception(192,  64,  96, 128, 16, 32, 32)
    local b3 = inception(256, 128, 128, 192, 32, 96, 64)

    net:add(a3):add(b3)
    net:add(MaxP(3,3,2,2,1,1))

    local a4 = inception(480, 192,  96, 208, 16,  48,  64)
    local b4 = inception(512, 160, 112, 224, 24,  64,  64)
    local c4 = inception(512, 128, 128, 256, 24,  64,  64)
    local d4 = inception(512, 112, 144, 288, 32,  64,  64)
    local e4 = inception(528, 256, 160, 320, 32, 128, 128)

    net:add(a4):add(b4):add(c4):add(d4):add(e4)
    net:add(MaxP(3,3,2,2,1,1))

    local a5 = inception(832, 256, 160, 320, 32, 128, 128)
    local b5 = inception(832, 384, 192, 384, 48, 128, 128)

    net:add(a5):add(b5)
    net:add(AvgP(8,8,1,1))
    net:add(nn.View(1024):setNumInputDims(3))
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(1024,10))
    return net
end
