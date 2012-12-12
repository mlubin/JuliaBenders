# type of data we get back from subproblems
const CutType = Real
# type of data we pass to subproblems
const SolType = Real

type MasterData
    cuts::Vector{CutType}
    sol::SolType
    status::String
    niter::Int
end

function MasterData()
    MasterData(CutType[],0.,"Not Done",0)
end

function masterproblem(data::MasterData)
    data.sol = mean(data.cuts)
    data.niter += 1
    if data.niter >= 5
        data.status = "Done"
    end
end


function subproblem(data)
    sleep(10*rand())
    return rand()
end

