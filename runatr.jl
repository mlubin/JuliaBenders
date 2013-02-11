require("trserial")

# asynchronous trust region (ATR) method of Linderoth and Wright


# asyncparam -- wait for this proportion of scenarios back before we resolve
function solveATR(nscen::Int, asyncparam::Float64, blocksize::Int, maxbasket::Int)
    const tr_max = 1000.
    const xi = 1e-4 # parameter related to accepting new iterate

    scenarioData = monteCarloSample(probdata,1:nscen)

    clpmaster = ClpModel()
    options = ClpSolve()
    set_presolve_type(options,1)
    set_log_level(clpmaster,0)
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

    # Current incumbent objective and solution index
    Qmin = [Inf,1]
    # list of candidate solutions
    candidates = Array(Vector{Float64},0)
    # corresponding (partial) objective values
    candidateQ = Array(Float64,0)
    # number of scenario solutions we've received back so far
    scenariosback = Array(Int,0)
    # if solution has triggered a reevaluation of the master 
    triggerednext = Array(Bool,0)
    # Incumbent solution when this solution was generated
    parentincumbent = Array(Int,0)
    # TR radius used when solving for this solution
    parentradius = Array(Float64,0)
    # model objective when this candidate was generated
    modelobj = Array(Float64,0)
    Tx = Array(Vector{Float64},0)
    tasks = Array((Int,Int),0)

    tr_radius = Inf
    update_tr(val) = (tr_radius = val)
    get_tr() = (tr_radius)

    tr_counter = 0
    update_counter(val) = (tr_counter = val)
    get_counter() = (tr_counter)

    basketsize() = sum(!(scenariosback .== nscen))

    function newcandidate(cand)
        push!(candidates,cand)
        push!(scenariosback,0)
        push!(triggerednext,false)
        push!(Tx,probdata.Tmat*cand)
        push!(candidateQ,dot(probdata.firstStageData.obj,cand))
        push!(parentincumbent,Qmin[2])
        push!(parentradius,get_tr())
        push!(modelobj,NaN)
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

    tr_radius = max(1,0.2*norm(candidates[1],Inf))

    

    converged = false
    set_converged() = (converged = true)
    is_converged() = (converged)
    niter = 0
    mastertime = 0.
    increment_mastertime(t) = (mastertime += t)
    @sync for p in 1:np
        if p != myid() || np == 1
            @async while !is_converged()
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
                    parentidx = parentincumbent[cand]
                    scenariosback[cand] += 1
                    candidateQ[cand] += optval/nscen
                    addCut(clpmaster,optval,subgrad,candidates[cand],s)
                    gennew = false
                    if scenariosback[cand] == nscen
                        gennew = true
                        if parentidx == 1 
                            if candidateQ[cand] < Qmin[1]
                                Qmin[1] = candidateQ[cand]
                                Qmin[2] = cand
                            end
                            println("Finished candidate $cand (val $(candidateQ[cand]), incumbent $(Qmin[1]), parent 1")
                        else
                            @assert scenariosback[parentidx] == nscen
                            parentQ = candidateQ[parentidx]
                            modelval = modelobj[parentidx]
                            println("Finished candidate $cand (val $(candidateQ[cand]), incumbent $(Qmin[1]), parent $parentQ, modelval $modelval, parent $parentidx")
                            
                            if candidateQ[cand] < Qmin[1] && 
                                candidateQ[cand] <= parentQ - xi*(parentQ - modelval)
                                # new incumbent
                                Qmin[1] = candidateQ[cand]
                                Qmin[2] = cand
                                println("Accepted new incumbent, objective value $(Qmin[1])")
                                # possibly increase trust region radius
                                if candidateQ[cand] <= parentQ - 0.5(parentQ-modelval) && 
                                    norm(candidates[cand]-candidates[parentidx],Inf) > parentradius[parentidx] - 1e-6
                                    update_tr(max(get_tr(),min(tr_max,2parentradius[parentidx])))
                                    println("Enlarged trust region radius to $(get_tr())")
                                end
                            else
                                # possibly reduce trust region radius
                                rho = min(1,parentradius[parentidx])*(candidateQ[cand]-parentQ)/(parentQ-modelval)
                                if rho > 0
                                    update_counter(get_counter()+1)
                                end
                                if rho > 3 || (get_counter() >= 3 && 1 < rho <= 3)
                                    update_tr(min(get_tr(),parentradius[parentidx]/min(rho,4)))
                                    println("Decreased trust region radius to $(get_tr())")
                                    update_counter(0)
                                end
                            end
                        end
                    end
                    if scenariosback[cand] >= asyncparam*nscen && !triggerednext[cand] && basketsize() < maxbasket
                        triggerednext[cand] = true
                        gennew = true
                    end
                    if gennew
                        # resolve master
                        setTR(clpmaster,candidates[cand],d.firstStageData.collb,d.firstStageData.colub,get_tr())
                        t = time()
                        initial_solve(clpmaster,options)
                        increment_mastertime(time()-t)
                        @assert is_proven_optimal(clpmaster)
                        # check convergence
                        if Qmin[1] - get_obj_value(clpmaster) < 1e-5(1+abs(Qmin[1]))
                            set_converged()
                            break
                        end
                        newcandidate(get_col_solution(clpmaster)[1:ncol1])
                        modelobj[end] = get_obj_value(clpmaster) 
                    end
                end
            end
        end
    end
    
    @assert converged
    println("Optimal objective is: $(Qmin[1]), $(length(candidates)) candidates ($(sum(scenariosback .== nscen)) complete), $(sum(scenariosback)) scenario subproblems solved")
    println("Time in master: $mastertime sec")

end

if length(ARGS) != 5 
    error("usage: runatr.jl [dataname] [num scenarios] [async param] [block size] [max basket]")
end

s = ARGS[1]
nscen = int(ARGS[2])
asyncparam = float(ARGS[3])
blocksize = int(ARGS[4])
maxbasket = int(ARGS[5])
d = SMPSData(string(s,".cor"),string(s,".tim"),string(s,".sto"),string(s,".sol"))
for p in 1:nprocs()
    remote_call_fetch(p,setGlobalProbData,d)
end
@time solveATR(nscen,asyncparam,blocksize,maxbasket)
