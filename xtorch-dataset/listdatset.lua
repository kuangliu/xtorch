--------------------------------------------------------------------------------
-- listdataset loads training/test data with list files containing the names
-- and targets.
--------------------------------------------------------------------------------

dofile('./xtorch-dataset/dataloader/listdataloader.lua')

local ListDataset = torch.class 'ListDataset'
pathcat = paths.concat

---------------------------------------------------------------
-- ListDataset takes params:
--  - directory: directory containing the images
--  - list: list file containg the image names
--  - imsize: image target size
--
-- other optional params:
--  - scale: input scale
--  - mean: optional, explicit data mean
--  - std: optional, explicit data std
--  - transform: table representing image processing functions
--     - standardize: perform zero mean and std normalization
--     - ...
--
function ListDataset:__init(opt)
    -- parse args
    for k,v in pairs(opt) do
        self[k] = v
    end
    -- init data loader
    self.dataloader = ListDataLoader(self.directory, self.list, self.imsize)
    self.N = self.dataloader.N
end

---------------------------------------------------------------
-- perform stardard transform including scale, zero mean
-- and std normalization
--
function ListDataset:__standardize(inputs)
    -- scale
    if self.scale then inputs:mul(self.scale) end

    -- zero mean & std normalization
    if not self.mean or not self.std then
        self:calcMeanStd()
    end

    for i = 1,3 do
        inputs[{ {},{i},{},{} }]:add(-self.mean[i]):div(self.std[i])
    end
    return inputs
end

---------------------------------------------------------------
-- calculate mean & std
--
function ListDataset:calcMeanStd()
    -- zero mean & std normalization
    if not self.mean or not self.std then
        print('==> computing mean & std..')
        local N = math.min(10000, self.dataloader.N)
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
-- perform a series of image processing functions on inputs
--
function ListDataset:__imfunc(inputs)
    -- no transform needed, return the raw inputs
    if not self.transform then return inputs end

    if self.transform.standardize then
        inputs = self:__standardize(inputs)
    end
    return inputs
end

---------------------------------------------------------------
-- load training batch sample
--
function ListDataset:sample(quantity)
    local inputs, targets = self.dataloader:sample(quantity)
    inputs = self:__imfunc(inputs)
    return inputs, targets
end

---------------------------------------------------------------
-- load test batch sample
--
function ListDataset:get(i1,i2)
    local inputs, targets = self.dataloader:get(i1,i2)
    inputs = self:__imfunc(inputs)
    return inputs, targets
end
