--------------------------------------------------------------------------------
-- plaindataset: wraps X & Y to give a unified interface.
--------------------------------------------------------------------------------

local PlainDataset = torch.class 'PlainDataset'

function PlainDataset:__init(X,Y)
    self.X = X
    self.Y = Y
    self.N = X:size(1)
end

---------------------------------------------------------------
-- load training batch sample
--
function PlainDataset:sample(quantity)
    local indices = torch.LongTensor(quantity):random(self.N)
    local samples = self.X:index(1, indices)
    local targets = self.Y:index(1, indices)
    return samples, targets
end

---------------------------------------------------------------
-- load test batch sample
--
function PlainDataset:get(i1,i2)
    local indices = torch.range(i1,i2):long()
    local samples = self.X:index(1, indices)
    local targets = self.Y:index(1, indices)
    return samples, targets
end
