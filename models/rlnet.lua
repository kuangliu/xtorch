require 'nn';

function getRLNet()
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(3,128,5,5,1,1,2,2))
    -- net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU(true))

    net:add(nn.SpatialConvolution(128,512,3,3,1,1,1,1))
    -- net:add(nn.SpatialBatchNormalization(512))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.125))
    net:add(nn.SpatialMaxPooling(2,2,2,2))

    net:add(nn.SpatialConvolution(512,128,1,1,1,1,0,0))
    -- net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.25))

    net:add(nn.SpatialConvolution(128,128,5,5,1,1,2,2))
    -- net:add(nn.SpatialBatchNormalization(128))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialMaxPooling(3,3,2,2))
    net:add(nn.Dropout(0.375))

    net:add(nn.SpatialConvolution(128,512,3,3,1,1,1,1))
    -- net:add(nn.SpatialBatchNormalization(512))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialAveragePooling(7,7,1,1))
    net:add(nn.View(-1):setNumInputDims(3))
    net:add(nn.Linear(512, 10))
    return net
end
