require("bendersserial")

# asynchronous l-shaped method (ALS) of Linderoth and Wright


# asyncparam -- wait for this proportion of scenarios back before we resolve
function solveBendersParallel(nscen::Int, asyncparam::Float64, blocksize::Int)

    scenarioData = monteCarloSample(probdata,1:nscen)
    np = nprocs()
    lpmasterproc = (np > 1) ? 2 : 1

    @everywhere begin
        ncol1 = probdata.firstStageData.ncol
        nrow1 = probdata.firstStageData.nrow
        nrow2 = probdata.secondStageTemplate.nrow
    end

    @spawnat lpmasterproc begin
        global clpmaster, options, probdata
        clpmaster = ClpModel()
        options = ClpSolve()
        set_presolve_type(options,1)

        # add \theta variables for cuts
        thetaidx = [(ncol1+1):(ncol1+nscen)]
        load_problem(clpmaster, probdata.Amat, probdata.firstStageData.collb,
            probdata.firstStageData.colub, probdata.firstStageData.obj, 
            probdata.firstStageData.rowlb, probdata.firstStageData.rowub)
        zeromat = SparseMatrixCSC(int32(nrow1),int32(nscen),ones(Int32,nscen+1),Int32[],Float64[])
        add_columns(clpmaster, -1e8*ones(nscen), Inf*ones(nscen),
            (1/nscen)*ones(nscen), zeromat)
    end


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
    if length(probdata.initialSolution) != 0
        println("Using provided starting solution")
        newcandidate(probdata.initialSolution)
    else
        newcandidate(solveExtensive(probdata,1))
    end

    converged = false
    set_converged() = (converged = true)
    is_converged() = (converged)
    niter = 0
    mastertime = 0.
    increment_mastertime(t) = (mastertime += t)
    @sync for p in 1:np
        if (p != myid() && p != lpmasterproc) || np == 1
            @async while !is_converged()
                mytasks = delete!(tasks,1:min(blocksize,length(tasks)))
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
                    f = @spawnat lpmasterproc (global clpmaster; addCut(clpmaster,optval,subgrad,candidates[cand],s))
                    #wait(f) # we don't need to wait
                    if scenariosback[cand] == nscen
                        if candidateQ[cand] < Qmin[1]
                            Qmin[1] = candidateQ[cand]
                            Qmin[2] = cand
                        end
                    end
                    if scenariosback[cand] >= asyncparam*nscen && !triggerednext[cand]
                        triggerednext[cand] = true
                        f = @spawnat lpmasterproc begin # resolve master
                            global clpmaster, options
                            x = time()
                            initial_solve(clpmaster,options)
                            mtime = time()-x
                            @printf("%.2f sec in master",mtime)
                            @assert is_proven_optimal(clpmaster)
                            mtime,get_obj_value(clpmaster), get_col_solution(clpmaster)[1:ncol1]
                        end
                        t, objval, colsol = fetch(f)
                        increment_mastertime(t)
                        # check convergence
                        if Qmin[1] - objval < 1e-5(1+abs(Qmin[1]))
                            set_converged()
                            break
                        end
                        newcandidate(colsol)
                    end
                end
            end
        end
    end
    
    @assert converged
    println("Optimal objective is: $(Qmin[1]), $(length(candidates)) candidates ($(sum(scenariosback .== nscen)) complete), $(sum(scenariosback)) scenario subproblems solved")
    println("Time in master: $mastertime sec")

end

if length(ARGS) != 4 
    error("usage: runatr.jl [dataname] [num scenarios] [async param] [block size]")
end

s = ARGS[1]
nscen = int(ARGS[2])
asyncparam = float(ARGS[3])
blocksize = int(ARGS[4])
d = SMPSData(string(s,".cor"),string(s,".tim"),string(s,".sto"),string(s,".sol"))
for p in 1:nprocs()
    remote_call_fetch(p,setGlobalProbData,d)
end
@time solveBendersParallel(nscen,asyncparam,blocksize)
