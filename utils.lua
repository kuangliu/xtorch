----------------------------------------------------------------------
-- Collection of useful functions for Torch.
--  - progress: progress bar with loss & accuracy output
--  - MSRinit: init layer weights
--  - makeDataParallelTable: enable multi-GPU
--  - log: auto history log
--  - saveCheckpoint: save checkpoint
--  - loadCheckpoint: load checkpoint
--

local utils = {}
----------------------------------------------------------------------
-- time
--
function utils.formatTime(seconds)
   -- decompose:
   local floor = math.floor
   local days = floor(seconds / 3600/24)
   seconds = seconds - days*3600*24
   local hours = floor(seconds / 3600)
   seconds = seconds - hours*3600
   local minutes = floor(seconds / 60)
   seconds = seconds - minutes*60
   local secondsf = floor(seconds)
   seconds = seconds - secondsf
   local millis = floor(seconds*1000)

   -- string
   local f = ''
   local i = 1
   if days > 0 then f = f .. days .. 'D' i=i+1 end
   if hours > 0 and i <= 2 then f = f .. hours .. 'h' i=i+1 end
   if minutes > 0 and i <= 2 then f = f .. minutes .. 'm' i=i+1 end
   if secondsf > 0 and i <= 2 then f = f .. secondsf .. 's' i=i+1 end
   if millis > 0 and i <= 2 then f = f .. millis .. 'ms' i=i+1 end
   if f == '' then f = '0ms' end

   -- return formatted time
   return f
end
local formatTime = xlua.formatTime

----------------------------------------------------------------------
-- progress bar
-- modified from xlua to give more information output.
--
do
   local function getTermLength()
      if sys.uname() == 'windows' then return 80 end
      local tputf = io.popen('tput cols', 'r')
      local w = tonumber(tputf:read('*a'))
      local rc = {tputf:close()}
      if rc[3] == 0 then return w
      else return 80 end
   end

   local barDone = true
   local previous = -1
   local tm = ''
   local timer
   local times
   local indices
   local termLength = math.min(getTermLength(), 120)
   function utils.progress(current, goal, loss, acc)
      -- defaults:
      local barLength = termLength - 54 --34 make bar shorter
      local smoothing = 100
      local maxfps = 10

      -- Compute percentage
      local percent = math.floor(((current) * barLength) / goal)

      -- start new bar
      if (barDone and ((previous == -1) or (percent < previous))) then
         barDone = false
         previous = -1
         tm = ''
         timer = torch.Timer()
         times = {timer:time().real}
         indices = {current}
      else
         io.write('\r')
      end

      --if (percent ~= previous and not barDone) then
      if (not barDone) then
         previous = percent
         -- print bar
         io.write(' [')
         for i=1,barLength do
            if (i < percent) then io.write('=')
            elseif (i == percent) then io.write('>')
            else io.write('.') end
         end
         io.write('] ')
         for i=1,termLength-barLength-4 do io.write(' ') end
         for i=1,termLength-barLength-4 do io.write('\b') end
         -- time stats
         local elapsed = timer:time().real
         local step = (elapsed-times[1]) / (current-indices[1])
         if current==indices[1] then step = 0 end
         local remaining = math.max(0,(goal - current)*step)
         table.insert(indices, current)
         table.insert(times, elapsed)
         if #indices > smoothing then
            indices = table.splice(indices)
            times = table.splice(times)
         end
         -- Print remaining time when running or total time when done.
         if (percent < barLength) then
            tm = ' ETA: ' .. formatTime(remaining)
         else
            tm = ' Tot: ' .. formatTime(elapsed)
         end
         tm = tm .. ' | Step: ' .. formatTime(step)
         io.write(tm)

         -- print loss & accuracy. acc could be abscent (for regression problems)
         acc = acc or 0
         io.write(string.format(' | loss: %.5f | acc: %.5f', loss, acc))


         -- go back to center of bar, and print progress
         for i=1,37+#tm+barLength/2 do io.write('\b') end
         io.write(' ', current, '/', goal, ' ')
         -- reset for next bar
         if (percent == barLength) then
            barDone = true
            io.write('\n')
         end
         -- flush
         io.write('\r')
         io.flush()
      end
   end
end

----------------------------------------------------------------
-- init layer weights
--
function utils.MSRinit(net)
    -- init CONV layer
    local function initconv(name)
        for _,layer in pairs(net:findModules(name)) do
            local n = layer.kW*layer.kH*layer.nOutputPlane
            layer.weight:normal(0,math.sqrt(2/n))
            if cudnn.version >= 4000 then
                layer.bias = nil
                layer.gradBias = nil
            else
                layer.bias:zero()
            end
        end
    end

    -- init BN layers
    local function initbn(name)
        for _,layer in pairs(net:findModules(name)) do
            layer.weight:fill(1)
            layer.bias:zero()
        end
    end

    initconv('cudnn.SpatialConvolution')
    initconv('nn.SpatialConvolution')
    initbn('cudnn.SpatialBatchNormalization')
    initbn('nn.SpatialBatchNormalization')

    -- init FC layers
    for _,layer in pairs(net:findModules'nn.Linear') do
        layer.bias:zero()
    end

    return net
end

----------------------------------------------------------------
-- enable multi-GPU
--
function utils.makeDataParallelTable(net, nGPU)
    if nGPU > 1 then
        local gpus = torch.range(1, nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
                    :add(net, gpus)
                    :threads(function()
                        local cudnn = require 'cudnn'
                        cudnn.fastest, cudnn.benchmark = fastest, benchmark
                    end)
        net = dpt:cuda()
    end
    return net
end

----------------------------------------------------------------
-- log
-- automatically create new log when training begins.
-- no specific log file needed.
--
function utils.addlog(...)
    paths.mkdir('log')
    -- get history logPath or create a new one named after the current time
    logPath = logPath or './log/'..sys.fexecute('date +"%Y-%m-%d-%H-%M-%S"')
    local f = io.open(logPath, 'a')
    for _,v in pairs({...}) do
        if type(v) == 'number' then v = ('%.4f'):format(v) end
        f:write(v..'\t')
    end
    f:write('\n')
    f:flush()
    f:close()
end

-------------------------------------------------------------------
-- save checkpoint
--
function utils.saveCheckpoint(net, epoch, optimState, bestAcc)
    paths.mkdir('checkpoint')
    local cpt = './checkpoint/'
    local modelfile = paths.concat(cpt, 'model.t7')
    local optimfile = paths.concat(cpt, 'optimState.t7')
    local latest = paths.concat(cpt, 'latest.t7')

    torch.save(modelfile, net)
    torch.save(optimfile, optimState)
    torch.save(latest, {
        epoch = epoch,
        modelfile = modelfile,
        optimfile = optimfile,
        bestAcc = bestAcc
    })
end

----------------------------------------------------------------
-- load checkpoint
--
function utils.loadCheckpoint()
    local latestPath = './checkpoint/latest.t7'
    assert(paths.filep(latestPath), 'Latest checkpoint not exist!')
    return torch.load(latestPath)
end

----------------------------------------------------------------
-- merge table
--
function utils.merge(A, B)
    for k,v in pairs(B) do
        A[k] = v
    end
    return A
end

return utils
