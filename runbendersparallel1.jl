load("bendersserial")



function solveBendersParallel(nscen::Integer)

    scenarioData = monteCarloSample(probdata,1:nscen)

    stage1sol = solveExtensive(probdata,1)
    
    clpmaster = ClpModel()
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

    thetasol = -1e8*ones(nscen)

    converged = false
    niter = 0
    mastertime = 0.
    const blocksize = 2
    while true
        Tx = d.Tmat*stage1sol
        # solve benders subproblems
        nviolated = 0
        #print("current solution is [")
        #for i in 1:ncol1
        #    print("$(stage1sol[i]),")
        #end
        #println("]")
        scen_count = 1
        next_block() = (idx=scen_count; scen_count+=blocksize; idx:min(idx+blocksize-1,nscen))
        new_violated() = (nviolated += 1)
        @sync for p in 1:np
            if p != myid() || np == 1
                @spawnlocal while true
                    scenblock = next_block()
                    if length(scenblock) == 0
                        break
                    end
                    results = remote_call_fetch(p,solveSubproblems,
                        [scenarioData[s][1]-Tx for s in scenblock],
                        [scenarioData[s][2]-Tx for s in scenblock])
                    for (s,result) in zip(scenblock,results)
                        optval, subgrad = result
                        #println("For scen $s, optval is $optval and model value is $(thetasol[s])")
                        if (optval > thetasol[s] + 1e-7)
                            new_violated()
                            addCut(clpmaster, optval, subgrad, stage1sol, s)
                        end
                    end
                end
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

s = ARGS[1]
nscen = int(ARGS[2])
d = SMPSData(strcat(s,".cor"),strcat(s,".tim"),strcat(s,".sto"))
for p in 1:nprocs()
    remote_call_fetch(p,setGlobalProbData,d)
end
@time solveBendersParallel(nscen)
