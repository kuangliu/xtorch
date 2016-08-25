--------------------------------------------------------------------------------
-- datagen wraps dataloader and provides a series of image processing and
-- augumentations function including:
--  - standardize: zero mean & std normalization
--  - randomflip
--  - randomcrop
--  - TODO: zca: zca whitening
--------------------------------------------------------------------------------

local DataGen = torch.class 'DataGen'
local pathcat = paths.concat

---------------------------------------------------------------
-- DataGen takes params:
--  - dataloader: to sample/get data
--  - standardize: zero mean and std normalization
--      - scale: input scale
--      - mean: input mean
--      - std: input std
--  - randomflip: random horizontal flip
--  - randomcrop: random input crop
--      - size: crop size
--      - pad: zero padding
--
function DataGen:__init(opt)
    local parse = function(A,B) for k,v in pairs(B) do A[k] = v end end
    -- parse {'dataloader', 'standardize', 'randomflip', 'randomcrop'}
    parse(self, opt)
    self.N = self.dataloader.N
    -- parse {'scale', 'mean', 'std'}
    if type(self.standardize) == 'table' then parse(self, self.standardize) end
    -- parse {'size', 'pad'}
    if type(self.randomcrop) == 'table' then parse(self, self.randomcrop) end
end

---------------------------------------------------------------
-- return mean & std
--
function DataGen:getmeanstd()
    -- zero mean & std normalization
    if self.mean and self.std then return self.mean, self.std end

    print('==> computing mean & std..')
    local N = math.min(10000, self.N)
    self.mean = torch.zeros(3)
    self.std = torch.zeros(3)
    for i = 1,N do
        xlua.progress(i,N)
        local im = self.dataloader:sample(1)[1]
        for j = 1,3 do
            self.mean[j] = self.mean[j] + im[j]:mean()
            self.std[j] = self.std[j] + im[j]:std()
        end
    end
    self.mean:div(N)
    self.std:div(N)
    return self.mean, self.std
end

---------------------------------------------------------------
-- perform stardard transform including scale, zero mean
-- and std normalization
function DataGen:__standardize(inputs)
    -- scale
    if self.scale then inputs:mul(self.scale) end

    -- zero mean & std normalization
    if not self.mean or not self.std then
        self:getmeanstd()
    end

    for i = 1,3 do
        inputs[{ {},{i},{},{} }]:add(-self.mean[i]):div(self.std[i])
    end
    return inputs
end

---------------------------------------------------------------
-- random horizontal flip
--
function DataGen:__randomflip(inputs)
    local batchSize = inputs:size(1)
    local flipMask = torch.randperm(batchSize):le(batchSize/2)

    for i = 1, batchSize do
        if flipMask[i] == 1 then
            image.hflip(inputs[i], inputs[i])
        end
    end
    return inputs
end

---------------------------------------------------------------
-- random crop with zero padding
--
function DataGen:__randomcrop(inputs)
    assert(inputs:dim() == 4, '[ERROR] random crop input size error!')
    local N,C,H,W = table.unpack(inputs:size():totable())
    local size = self.size or math.min(H, W)   -- outputs sized [N,C,size,size]
    local P = self.pad or 4                    -- pad 4 zeros by default
    assert(size <= H+2*P and size <= W+2*P, '[ERROR] crop size is too large!')

    local padded = torch.zeros(N, C, H+2*P, W+2*P) -- padded sized [N,C,H+2*P,W+2*P]
    padded:narrow(4,1+P,W):narrow(3,1+P,H):copy(inputs)
    local x = torch.random(1,1+W+2*P-size)
    local y = torch.random(1,1+H+2*P-size)
    return padded:narrow(4,x,size):narrow(3,y,size)
end


---------------------------------------------------------------
-- perform a series of image processing functions on inputs
--
function DataGen:__imfunc(inputs)
    if self.standardize then inputs = self:__standardize(inputs) end
    if self.randomflip then inputs = self:__randomflip(inputs) end
    if self.randomcrop then inputs = self:__randomcrop(inputs) end
    return inputs
end

---------------------------------------------------------------
-- load training batch sample
--
function DataGen:sample(quantity)
    local inputs, targets = self.dataloader:sample(quantity)
    inputs = self:__imfunc(inputs)
    return inputs, targets
end

---------------------------------------------------------------
-- load test batch sample
--
function DataGen:get(i1,i2)
    local inputs, targets = self.dataloader:get(i1,i2)
    inputs = self:__imfunc(inputs)
    return inputs, targets
end
