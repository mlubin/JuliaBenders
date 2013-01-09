load("bendersserial")

# asynchronous l-shaped method (ALS) of Linderoth and Wright


# asyncparam -- wait for this proportion of scenarios back before we resolve
function solveBendersParallel(nscen::Integer,asyncparam::Float64)

    scenarioData = monteCarloSample(probdata,1:nscen)

    stage1sol = solveExtensive(probdata,1)
    
    clpmaster = ClpModel()
    options = ClpSolve()
    set_presolve_type(options,1)
    np = nprocs()

    ncol1 = probdata.firstStageData.ncol
    nrow1 = probdata.firstStageData.nrow
    nrow2 = probdata.secondStageTemplate.nrow
    # add \theta variables for cuts
    thetaidx = [(ncol1+1):(ncol1+nscen)]
    load_problem(clpmaster, probdata.Amat, probdata.firstStageData.collb,
        probdata.firstStageData.colub, probdata.firstStageData.obj, 
        probdata.firstStageData.rowlb, probdata.firstStageData.rowub)
    zeromat = SparseMatrixCSC(int32(nrow1),int32(nscen),ones(Int32,nscen+1),Int32[],Float64[])
    add_columns(clpmaster, -1e8*ones(nscen), Inf*ones(nscen),
        (1/nscen)*ones(nscen), zeromat)

    @everywhere load_problem(clpsubproblem, probdata.Wmat, probdata.secondStageTemplate.collb,
        probdata.secondStageTemplate.colub, probdata.secondStageTemplate.obj,
        probdata.secondStageTemplate.rowlb, probdata.secondStageTemplate.rowub)

    # (Qminval,QminIdx)
    Qmin = [Inf,0]
    candidates = Array(Vector{Float64},0)
    candidateQ = Array(Float64,0)
    scenariosback = Array(Int,0)
    triggerednext = Array(Bool,0)
    Tx = Array(Vector{Float64},0)
    tasks = Array((Int,Int),0)

    function newcandidate(cand)
        push!(candidates,cand)
        push!(scenariosback,0)
        push!(triggerednext,false)
        push!(Tx,probdata.Tmat*cand)
        push!(candidateQ,dot(probdata.firstStageData.obj,cand))
        for i in 1:nscen
            push!(tasks,(length(candidates),i))
        end
    end
    
    # initial guess
    newcandidate(solveExtensive(probdata,1))

    converged = false
    set_converged() = (converged = true)
    is_converged() = (converged)
    niter = 0
    mastertime = 0.
    increment_mastertime(t) = (mastertime += t)
    const blocksize = 2
    @sync for p in 1:np
        if p != myid() || np == 1
            @spawnlocal while !is_converged()
                mytasks = tasks[1:min(blocksize,length(tasks))]
                for i in 1:length(mytasks) # TODO: improve syntax
                    shift!(tasks)
                end
                if length(mytasks) == 0
                    yield()
                    continue
                end
                results = remote_call_fetch(p,solveSubproblems,
                    [scenarioData[s][1]-Tx[cand] for (cand,s) in mytasks],
                    [scenarioData[s][2]-Tx[cand] for (cand,s) in mytasks])
                for i in 1:length(mytasks)
                    cand,s = mytasks[i]
                    optval, subgrad = results[i]
                    scenariosback[cand] += 1
                    candidateQ[cand] += optval/nscen
                    addCut(clpmaster,optval,subgrad,candidates[cand],s)
                    if scenariosback[cand] == nscen
                        if candidateQ[cand] < Qmin[1]
                            Qmin[1] = candidateQ[cand]
                            Qmin[2] = cand
                        end
                    end
                    if scenariosback[cand] >= asyncparam*nscen && !triggerednext[cand]
                        triggerednext[cand] = true
                        # resolve master
                        t = time()
                        initial_solve(clpmaster,options)
                        increment_mastertime(time()-t)
                        @assert is_proven_optimal(clpmaster)
                        # check convergence
                        if Qmin[1] - get_obj_value(clpmaster) < 1e-7(1+abs(Qmin[1]))
                            set_converged()
                            break
                        end
                        newcandidate(get_col_solution(clpmaster)[1:ncol1])
                    end
                end
            end
        end
    end
    
    @assert converged
    println("Optimal objective is: $(Qmin[1]), $(length(candidates)) candidates ($(sum(scenariosback .== nscen)) complete), $(sum(scenariosback)) scenario subproblems solved")
    println("Time in master: $mastertime sec")

end

s = ARGS[1]
nscen = int(ARGS[2])
asyncparam = float(ARGS[3])
d = SMPSData(strcat(s,".cor"),strcat(s,".tim"),strcat(s,".sto"))
for p in 1:nprocs()
    remote_call_fetch(p,setGlobalProbData,d)
end
@time solveBendersParallel(nscen,asyncparam)
