--------------------------------------------------------------------------------
-- datagen wraps dataloader and providing a series of image processing functions
-- including:
--  - zero mean & std normalization
--  - ...
--------------------------------------------------------------------------------

local DataGen = torch.class 'DataGen'
local pathcat = paths.concat

---------------------------------------------------------------
-- DataGen takes params:
--  - dataloader: to sample/get data
--  - transform: table representing image processing functions, like
--     - standardize: perform zero mean and std normalization
--     - ...
--
function DataGen:__init(opt)
    for k,v in pairs(opt) do self[k] = v end
    self.N = self.dataloader.N
end

---------------------------------------------------------------
-- return mean & std
--
function DataGen:getmeanstd()
    -- zero mean & std normalization
    if not self.mean or not self.std then
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
    end
    return self.mean, self.std
end

---------------------------------------------------------------
-- perform stardard transform including scale, zero mean
-- and std normalization
--
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
-- perform a series of image processing functions on inputs
--
function DataGen:__imfunc(inputs)
    if self.standardize then
        inputs = self:__standardize(inputs)
    end
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
