require("extensive")

const clpsubproblem = ClpModel()
set_log_level(clpsubproblem,0)

function setGlobalProbData(d::SMPSData)
    global probdata = d
end

function solveSubproblems(rowlbs, rowubs)
    return [solveSubproblem(rowlbs[i],rowubs[i]) for i in 1:length(rowlbs)]
end

function solveSubproblem(rowlb, rowub)

    Tmat = probdata.Tmat
    ncol1 = probdata.firstStageData.ncol
    nrow2 = probdata.secondStageTemplate.nrow

    chg_row_lower(clpsubproblem,rowlb)
    chg_row_upper(clpsubproblem,rowub)
    initial_solve(clpsubproblem)
    # don't handle infeasible subproblems yet
    @assert is_proven_optimal(clpsubproblem)
    optval = get_obj_value(clpsubproblem)
    duals = dual_row_solution(clpsubproblem)
    
    subgrad = zeros(ncol1)
    for i in 1:nrow2
        status = get_row_status(clpsubproblem,i)
        if (status == 1) # basic
            continue
        end
        for k in 1:ncol1
            subgrad[k] += -duals[i]*Tmat[i,k]
        end
    end

    return optval, subgrad
end

function addCut(master::ClpModel, optval::Float64, subgrad::Vector{Float64}, stage1sol::Vector{Float64}, scen)
    #print("adding cut: [")
    # add (0-based) cut to master
    cutvec = Float64[]
    cutcolidx = Int32[]
    for k in 1:length(subgrad)
     #   print("$(subgrad[k]),")
        if abs(subgrad[k]) > 1e-10
            push!(cutvec,-subgrad[k])
            push!(cutcolidx,k-1)
        end
    end
    #println("]")
    push!(cutvec,1.)
    push!(cutcolidx,length(subgrad)+scen-1)
    cutnnz = length(cutvec)
    cutlb = optval-dot(subgrad,stage1sol)

    add_rows(master, 1, [cutlb], [1e25], Int32[0,cutnnz], cutcolidx, cutvec)


end


function solveBendersSerial(d::SMPSData, nscen::Integer)

    scenarioData = monteCarloSample(d,1:nscen)

    stage1sol = solveExtensive(d,1)
    
    clpmaster = ClpModel()
    setGlobalProbData(d)
    ncol1 = d.firstStageData.ncol
    nrow1 = d.firstStageData.nrow
    nrow2 = d.secondStageTemplate.nrow
    # add \theta variables for cuts
    thetaidx = [(ncol1+1):(ncol1+nscen)]
    load_problem(clpmaster, d.Amat, d.firstStageData.collb,
        d.firstStageData.colub, d.firstStageData.obj, d.firstStageData.rowlb,
        d.firstStageData.rowub)
    zeromat = SparseMatrixCSC(int32(nrow1),int32(nscen),ones(Int32,nscen+1),Int32[],Float64[])
    add_columns(clpmaster, -1e8*ones(nscen), Inf*ones(nscen),
        (1/nscen)*ones(nscen), zeromat)

    load_problem(clpsubproblem, d.Wmat, d.secondStageTemplate.collb,
        d.secondStageTemplate.colub, d.secondStageTemplate.obj,
        d.secondStageTemplate.rowlb, d.secondStageTemplate.rowub)

    thetasol = -1e8*ones(nscen)

    converged = false
    niter = 0
    mastertime = 0.
    while true
        Tx = d.Tmat*stage1sol
        # solve benders subproblems
        nviolated = 0
        #print("current solution is [")
        #for i in 1:ncol1
        #    print("$(stage1sol[i]),")
        #end
        #println("]")
        for s in 1:nscen
            optval, subgrad = solveSubproblem(scenarioData[s][1]-Tx,scenarioData[s][2]-Tx)
            #println("For scen $s, optval is $optval and model value is $(thetasol[s])")
            if (optval > thetasol[s] + 1e-7)
                nviolated += 1
                addCut(clpmaster, optval, subgrad, stage1sol, s)
            end

        end

        if nviolated == 0
            break
        end
        println("Generated $nviolated violated cuts")
        # resolve master
        t = time()
        initial_solve(clpmaster)
        mastertime += time() - t
        @assert is_proven_optimal(clpmaster)
        sol = get_col_solution(clpmaster)
        stage1sol = sol[1:ncol1]
        thetasol = sol[(ncol1+1):end]
        niter += 1
    end

    println("Optimal objective is: $(get_obj_value(clpmaster)), $niter iterations")
    println("Time in master: $mastertime sec")

end

