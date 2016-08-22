--------------------------------------------------------------------------------
-- plaindataloader: loads X and Y that could be fit in memory.
--------------------------------------------------------------------------------

local PlainDataLoader = torch.class 'PlainDataLoader'

function PlainDataLoader:__init(opt)
    self.X = opt.X
    self.Y = opt.Y
    self.N = self.X:size(1)
end

---------------------------------------------------------------
-- load training batch sample
--
function PlainDataLoader:sample(quantity)
    local indices = torch.LongTensor(quantity):random(self.N)
    local samples = self.X:index(1, indices)
    local targets = self.Y:index(1, indices)
    return samples, targets
end

---------------------------------------------------------------
-- load test batch sample
--
function PlainDataLoader:get(i1,i2)
    local indices = torch.range(i1,i2):long()
    local samples = self.X:index(1, indices)
    local targets = self.Y:index(1, indices)
    return samples, targets
end
